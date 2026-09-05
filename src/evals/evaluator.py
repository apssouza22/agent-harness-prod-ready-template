"""Evaluator for evals."""

import time
from datetime import (
    datetime,
    timedelta,
)
from time import sleep

import openai
from langfuse import Langfuse
from langfuse.api.commons.types.observation_v2 import ObservationV2
from tqdm import tqdm

from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.llm.factory import build_openai_client_kwargs
from src.evals.helpers import (
    calculate_avg_scores,
    generate_report,
    get_input_output,
    initialize_metrics_summary,
    initialize_report,
    process_trace_results,
    update_failure_metrics,
    update_success_metrics,
)
from src.evals.metrics import metrics
from src.evals.schemas import ScoreSchema


class Evaluator:
    """Evaluates model outputs using predefined metrics.

    This class handles fetching root observations from Langfuse, evaluating them
    against metrics, and uploading scores back to Langfuse.

    Attributes:
        client: OpenAI client for API calls.
        langfuse: Langfuse client for trace management.
    """

    def __init__(self):
        """Initialize Evaluator with OpenAI and Langfuse clients."""
        self.client = openai.AsyncOpenAI(
            **build_openai_client_kwargs(
                api_key=settings.EVALUATION_API_KEY,
                base_url=settings.EVALUATION_BASE_URL,
                bifrost_agent="agent_1",
            )
        )
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            timeout=60,
        )
        self.report = initialize_report(settings.EVALUATION_LLM)
        initialize_metrics_summary(self.report, metrics)

    async def run(self, generate_report_file=True):
        """Main execution function that fetches and evaluates traces.

        Retrieves root observations from Langfuse, evaluates each one against all
        metrics, and uploads the scores back to Langfuse.

        Args:
            generate_report_file: Whether to generate a JSON report after evaluation. Defaults to True.
        """
        start_time = time.time()
        observations = self.__fetch_root_observations()
        self.report["total_traces"] = len(observations)

        trace_results = {}

        for observation in tqdm(observations, desc="Evaluating traces"):
            trace_id = observation.trace_id
            if not trace_id:
                continue

            trace_results[trace_id] = {
                "success": False,
                "metrics_evaluated": 0,
                "metrics_succeeded": 0,
                "metrics_results": {},
            }

            for metric in tqdm(metrics, desc=f"Applying metrics to trace {trace_id[:8]}...", leave=False):
                metric_name = metric["name"]
                input_text, output_text = get_input_output(observation)
                score = await self._run_metric_evaluation(metric, input_text, output_text)

                if score:
                    self._push_to_langfuse(trace_id, score, metric)
                    update_success_metrics(self.report, trace_id, metric_name, score, trace_results)
                else:
                    update_failure_metrics(self.report, trace_id, metric_name, trace_results)

                trace_results[trace_id]["metrics_evaluated"] += 1

            process_trace_results(self.report, trace_id, trace_results, len(metrics))
            sleep(settings.EVALUATION_SLEEP_TIME)

        self.report["duration_seconds"] = round(time.time() - start_time, 2)
        calculate_avg_scores(self.report)

        if generate_report_file:
            generate_report(self.report)

        logger.info(
            "Evaluation completed",
            total_traces=self.report["total_traces"],
            successful_traces=self.report["successful_traces"],
            failed_traces=self.report["failed_traces"],
            duration_seconds=self.report["duration_seconds"],
        )

    def _push_to_langfuse(self, trace_id: str, score: ScoreSchema, metric: dict):
        """Push evaluation score to Langfuse.

        Args:
            trace_id: The trace to score.
            score: The evaluation score.
            metric: The metric used for evaluation.
        """
        self.langfuse.create_score(
            trace_id=trace_id,
            name=metric["name"],
            data_type="NUMERIC",
            value=score.score,
            comment=score.reasoning,
        )

    async def _run_metric_evaluation(self, metric: dict, input: str, output: str) -> ScoreSchema | None:
        """Evaluate a single trace against a specific metric.

        Args:
            metric: The metric definition to use for evaluation.
            input: The input to evaluate.
            output: The output to evaluate.

        Returns:
            ScoreSchema with evaluation results or None if evaluation failed.
        """
        metric_name = metric["name"]
        if not metric:
            logger.error(f"Metric {metric_name} not found")
            return None
        system_metric_prompt = metric["prompt"]

        if not input or not output:
            logger.error(f"Metric {metric_name} evaluation failed", input=input, output=output)
            return None
        score = await self._call_openai(system_metric_prompt, input, output)
        if score:
            logger.info(f"Metric {metric_name} evaluation completed successfully", score=score)
        else:
            logger.error(f"Metric {metric_name} evaluation failed")
        return score

    async def _call_openai(self, metric_system_prompt: str, input: str, output: str) -> ScoreSchema | None:
        """Call OpenAI API to evaluate a trace.

        Args:
            metric_system_prompt: System prompt defining the evaluation metric.
            input: Formatted input messages.
            output: Formatted output message.

        Returns:
            ScoreSchema with evaluation results or None if API call failed.
        """
        num_retries = 3
        for _ in range(num_retries):
            try:
                response = await self.client.beta.chat.completions.parse(
                    model=settings.EVALUATION_LLM,
                    messages=[
                        {"role": "system", "content": metric_system_prompt},
                        {"role": "user", "content": f"Input: {input}\nGeneration: {output}"},
                    ],
                    response_format=ScoreSchema,
                )
                return response.choices[0].message.parsed
            except Exception as e:
                SLEEP_TIME = 10
                logger.error("Error calling OpenAI", error=str(e), sleep_time=SLEEP_TIME)
                sleep(SLEEP_TIME)
                continue
        return None

    def __fetch_root_observations(self) -> list[ObservationV2]:
        """Fetch root observations from the past 24 hours without scores.

        Returns:
            List of root observations whose traces have not been scored yet.
        """
        from_timestamp = datetime.now() - timedelta(hours=24)
        to_timestamp = datetime.now()
        logger.info("fetching_langfuse_observations", from_timestamp=str(from_timestamp))
        try:
            scored_trace_ids = self._get_scored_trace_ids(from_timestamp, to_timestamp)
            observations: list[ObservationV2] = []
            cursor = None

            while True:
                response = self.langfuse.api.observations.get_many(
                    from_start_time=from_timestamp,
                    to_start_time=to_timestamp,
                    is_root_observation=True,
                    fields="core,basic,io",
                    limit=100,
                    cursor=cursor,
                )
                observations.extend(response.data)
                cursor = response.meta.cursor if response.meta else None
                if not cursor:
                    break

            return [
                observation
                for observation in observations
                if observation.trace_id and observation.trace_id not in scored_trace_ids
            ]
        except Exception as e:
            logger.error("error_fetching_observations", error=str(e))
            return []

    def _get_scored_trace_ids(self, from_timestamp: datetime, to_timestamp: datetime) -> set[str]:
        scored_trace_ids: set[str] = set()
        cursor = None

        while True:
            response = self.langfuse.api.scores_v3.get_many_v3(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                fields="core",
                limit=100,
                cursor=cursor,
            )
            for score in response.data:
                subject = score.subject
                if getattr(subject, "kind", None) == "trace":
                    scored_trace_ids.add(subject.id)

            cursor = response.meta.cursor if response.meta else None
            if not cursor:
                break

        return scored_trace_ids
