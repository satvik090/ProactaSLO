import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from app.cache import MetricCache
from app.config import MODEL_DIR
from app.observability import model_drift_signal, slo_prediction_score

logger = logging.getLogger(__name__)

WINDOW_SIZE = 30
INPUT_SIZE = 8
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

cache = MetricCache()


class SLOPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.fc(output[:, -1, :])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.forward_logits(inputs))


class SLODataset(Dataset):
    def __init__(self, vectors: list[list[float]], labels: list[int]) -> None:
        self.vectors = [_first_eight(vector) for vector in vectors]
        self.labels = labels

    def __len__(self) -> int:
        return max(0, min(len(self.vectors), len(self.labels)) - WINDOW_SIZE)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.vectors[index : index + WINDOW_SIZE]
        label = self.labels[index + WINDOW_SIZE]
        return torch.tensor(window, dtype=torch.float32), torch.tensor([float(label)], dtype=torch.float32)


def train_model(service_name: str, vectors: list[list[float]], labels: list[int]) -> tuple[float, int]:
    dataset = SLODataset(vectors, labels)
    if len(dataset) < 2:
        logger.warning("Not enough samples to train %s; need more than %s vectors", service_name, WINDOW_SIZE)
        return float("inf"), 0

    positives = max(1, sum(1 for label in labels if label == 1))
    negatives = sum(1 for label in labels if label == 0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)

    split_index = max(1, int(len(dataset) * 0.8))
    if split_index >= len(dataset):
        split_index = len(dataset) - 1
    train_dataset = Subset(dataset, range(0, split_index))
    val_dataset = Subset(dataset, range(split_index, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SLOPredictor()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_vectors = torch.tensor([_first_eight(vector) for vector in vectors[: split_index + WINDOW_SIZE]], dtype=torch.float32)
    training_mean = training_vectors.mean(dim=0)
    training_std = training_vectors.std(dim=0, unbiased=False).clamp_min(1e-6)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epoch_count = 0

    for epoch in range(1, 101):
        epoch_count = epoch
        model.train()
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model.forward_logits(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    _checkpoint_path(service_name).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_state_dict": best_state,
            "training_mean": training_mean.tolist(),
            "training_std": training_std.tolist(),
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
        },
        _checkpoint_path(service_name),
    )
    return best_val_loss, epoch_count


async def predict(service_name: str) -> float:
    cached_score = await cache.get_prediction(service_name)
    if cached_score is not None:
        return cached_score

    checkpoint = _load_checkpoint(service_name)
    if checkpoint is None:
        logger.warning("No checkpoint found for %s; returning neutral prediction", service_name)
        return await _record_score(service_name, 0.5)

    vectors = await cache.read_metrics(service_name, WINDOW_SIZE)
    if len(vectors) < WINDOW_SIZE:
        logger.warning("Only %s vectors available for %s; returning neutral prediction", len(vectors), service_name)
        return await _record_score(service_name, 0.5)

    model = _model_from_checkpoint(checkpoint)
    model.eval()
    inputs = torch.tensor([[_first_eight(vector) for vector in vectors[-WINDOW_SIZE:]]], dtype=torch.float32)
    with torch.no_grad():
        score = torch.sigmoid(model.forward_logits(inputs)).item()

    return await _record_score(service_name, score)


async def compute_drift(service_name: str) -> float:
    checkpoint = _load_checkpoint(service_name)
    if checkpoint is None:
        logger.warning("No checkpoint found for %s; drift defaults to 0.0", service_name)
        return 0.0

    vectors = await cache.read_metrics(service_name, WINDOW_SIZE)
    if not vectors:
        logger.warning("No vectors available for %s; drift defaults to 0.0", service_name)
        return 0.0

    current = torch.tensor([_first_eight(vector) for vector in vectors[-WINDOW_SIZE:]], dtype=torch.float32)
    current_mean = current.mean(dim=0)
    train_mean = torch.tensor(checkpoint["training_mean"], dtype=torch.float32)
    train_std = torch.tensor(checkpoint["training_std"], dtype=torch.float32).clamp_min(1e-6)
    drift = torch.mean(torch.abs(current_mean - train_mean) / train_std).item()
    model_drift_signal.labels(service=service_name).set(drift)
    return drift


async def load_all_models(services: list[str]) -> None:
    available: list[str] = []
    missing: list[str] = []
    for service in services:
        if _checkpoint_path(service).exists():
            available.append(service)
        else:
            missing.append(service)
        await predict(service)
    logger.info("Loaded ProactaSLO checkpoints for: %s", ", ".join(available) if available else "none")
    logger.info("Missing ProactaSLO checkpoints for: %s", ", ".join(missing) if missing else "none")


def _evaluate(model: SLOPredictor, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            logits = model.forward_logits(batch_inputs)
            losses.append(float(loss_fn(logits, batch_labels).item()))
    return sum(losses) / len(losses) if losses else float("inf")


def _checkpoint_path(service_name: str) -> Path:
    return MODEL_DIR / f"{service_name}.pt"


def _load_checkpoint(service_name: str) -> dict[str, Any] | None:
    path = _checkpoint_path(service_name)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _model_from_checkpoint(checkpoint: dict[str, Any]) -> SLOPredictor:
    model = SLOPredictor(
        input_size=int(checkpoint.get("input_size", INPUT_SIZE)),
        hidden_size=int(checkpoint.get("hidden_size", HIDDEN_SIZE)),
        num_layers=int(checkpoint.get("num_layers", NUM_LAYERS)),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint["state_dict"]))
    return model


async def _record_score(service_name: str, score: float) -> float:
    await cache.set_prediction(service_name, score)
    slo_prediction_score.labels(service=service_name).set(score)
    return score


def _first_eight(vector: list[float]) -> list[float]:
    features = [float(value) for value in vector[:INPUT_SIZE]]
    if len(features) < INPUT_SIZE:
        features.extend([0.0] * (INPUT_SIZE - len(features)))
    return features


@dataclass
class Prediction:
    service: str
    risk: float
    projected_success_rate: float
    minutes_to_breach: float | None
    generated_at: float


class PredictionEngine:
    async def predict(self, service: str, samples: list[Any], target: float) -> Prediction:
        score = await predict(service)
        return Prediction(
            service=service,
            risk=score,
            projected_success_rate=max(0.0, 1.0 - score),
            minutes_to_breach=None,
            generated_at=time.time(),
        )
