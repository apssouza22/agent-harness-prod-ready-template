"""Prometheus metrics configuration for the application.

This module sets up and configures Prometheus metrics for monitoring the application.
"""

import os

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector
from starlette.requests import Request
from starlette.responses import Response


async def metrics(request: Request) -> Response:
    """Expose Prometheus metrics for scraping."""
    if "prometheus_multiproc_dir" in os.environ:
        registry = CollectorRegistry()
        MultiProcessCollector(registry)
    else:
        registry = REGISTRY

    return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})


def setup_metrics(app):
    """Set up Prometheus metrics endpoints.

    HTTP request metrics are recorded by MetricsMiddleware. starlette-prometheus
    middleware is not used because it is incompatible with FastAPI 0.137+ included
    routers (_IncludedRouter has no .path attribute).

    Args:
        app: FastAPI application instance
    """
    app.add_route("/metrics", metrics)
