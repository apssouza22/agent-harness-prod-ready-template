"""Prometheus metrics configuration for the application.

This module sets up and configures Prometheus metrics for monitoring the application.
"""

from prometheus_client import Counter, Histogram, Gauge
from starlette_prometheus import metrics, PrometheusMiddleware

# Request metrics
http_requests_total = Counter("http_requests_total", "Total number of HTTP requests", ["method", "endpoint", "status"])

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["method", "endpoint"]
)


# Database metrics
db_connections = Gauge("db_connections", "Number of active database connections")

# Custom business metrics
orders_processed = Counter("orders_processed_total", "Total number of orders processed")

llm_inference_duration_seconds = Histogram(
    "llm_inference_duration_seconds",
    "Time spent processing LLM inference",
    ["model", "agent_name"],
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
)

llm_stream_duration_seconds = Histogram(
    "llm_stream_duration_seconds",
    "Time spent processing LLM stream inference",
    ["model", "agent_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)


tokens_in_counter = Counter(
    "llm_tokens_in", "Number of input tokens", ["agent_name"]
)

tokens_out_counter = Counter(
    "llm_tokens_out", "Number of output tokens", ["agent_name"]
)

error_counter = Counter(
    "llm_errors", "Number of errors during LLM execution", ["agent_name"]
)

tool_executions_total = Counter(
    "tool_executions_total",
    "Total tool executions",
    ["tool_name", "status"]
)

# Guardrail metrics
guardrail_checks_total = Counter(
    "guardrail_checks_total",
    "Total guardrail checks executed",
    ["guardrail_type", "check_type", "result"],
)

guardrail_check_duration_seconds = Histogram(
    "guardrail_check_duration_seconds",
    "Duration of guardrail checks in seconds",
    ["guardrail_type", "check_type"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

guardrail_pii_detections_total = Counter(
    "guardrail_pii_detections_total",
    "Total PII detections by type",
    ["guardrail_type", "pii_type"],
)

guardrail_requests_blocked_total = Counter(
    "guardrail_requests_blocked_total",
    "Total requests blocked or modified by guardrails",
    ["guardrail_type", "reason"],
)
