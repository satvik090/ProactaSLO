import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aiohttp import web


SERVICES = [
    "auth",
    "payment",
    "order",
    "inventory",
    "shipping",
    "notification",
    "user",
    "cart",
    "search",
    "recommendation",
    "review",
    "warehouse",
    "gateway",
    "scheduler",
    "audit",
]

PORTS = dict(zip(SERVICES, range(8001, 8016)))
METRIC_BASELINES = {
    "p50_latency": (50.0, 5.0),
    "p95_latency": (120.0, 10.0),
    "p99_latency": (200.0, 20.0),
    "error_rate": (0.001, 0.0002),
    "request_rate": (100.0, 10.0),
    "cpu_util": (0.4, 0.05),
    "memory_usage": (0.5, 0.05),
    "queue_depth": (10.0, 2.0),
}
GATEWAY_CASCADE_SERVICES = {"order", "payment", "auth"}


@dataclass
class ServiceState:
    service: str
    violation_until: float = 0.0
    cascade_until: float = 0.0
    next_violation_at: float = field(default_factory=lambda: time.time() + random.randint(7200, 14400))

    def in_violation(self) -> bool:
        return time.time() < self.violation_until

    def in_gateway_cascade(self) -> bool:
        return time.time() < self.cascade_until


states = {service: ServiceState(service) for service in SERVICES}


async def metrics_handler(request: web.Request) -> web.Response:
    service = request.app["service"]
    body = render_metrics(service)
    return web.Response(text=body, headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"})


def render_metrics(service: str) -> str:
    state = states[service]
    lines: list[str] = []
    for metric_name in METRIC_BASELINES:
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f'{metric_name}{{job="{service}"}} {sample_metric(service, metric_name, state):.8f}')
    return "\n".join(lines) + "\n"


def sample_metric(service: str, metric_name: str, state: ServiceState) -> float:
    mean, std = METRIC_BASELINES[metric_name]
    if state.in_violation() and metric_name == "p99_latency":
        return max(0.0, random.gauss(600.0, 50.0))
    if state.in_violation() and metric_name == "error_rate":
        return max(0.0, random.gauss(0.05, 0.005))
    if service in GATEWAY_CASCADE_SERVICES and state.in_gateway_cascade() and metric_name == "p99_latency":
        return max(0.0, random.gauss(400.0, 40.0))
    return max(0.0, random.gauss(mean, std))


async def violation_controller(service: str) -> None:
    state = states[service]
    while True:
        await asyncio.sleep(max(1.0, state.next_violation_at - time.time()))
        duration = random.randint(600, 1200)
        state.violation_until = time.time() + duration
        timestamp = datetime.now(timezone.utc).isoformat()
        print(f"[VIOLATION INJECTED] {timestamp} service={service} duration={duration}s", flush=True)

        if service == "gateway":
            for cascade_service in GATEWAY_CASCADE_SERVICES:
                states[cascade_service].cascade_until = state.violation_until

        await asyncio.sleep(duration)
        state.violation_until = 0.0
        timestamp = datetime.now(timezone.utc).isoformat()
        print(f"[VIOLATION CLEARED] {timestamp} service={service}", flush=True)
        state.next_violation_at = time.time() + random.randint(7200, 14400)


async def start_server(service: str, port: int) -> None:
    app = web.Application()
    app["service"] = service
    app.router.add_get("/metrics", metrics_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"[STARTUP] service={service} port={port}", flush=True)
    await asyncio.Event().wait()


async def main() -> None:
    server_tasks = [start_server(service, port) for service, port in PORTS.items()]
    violation_tasks = [violation_controller(service) for service in SERVICES]
    await asyncio.gather(*server_tasks, *violation_tasks)


if __name__ == "__main__":
    asyncio.run(main())
