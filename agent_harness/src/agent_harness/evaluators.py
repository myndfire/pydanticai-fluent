"""Evaluators for agent output quality and safety."""

from typing import Protocol, Any
import structlog

from pydantic_ai import Agent


logger = structlog.get_logger()


class Evaluator(Protocol):
    """Protocol for evaluators."""

    async def evaluate(self, prompt: str, result: Any, context: dict) -> None:
        """
        Evaluate agent result.

        Args:
            prompt: User prompt
            result: Agent result
            context: Additional context (session_id, metadata, etc.)
        """
        ...


class QualityCheck:
    """LLM-as-judge quality evaluation."""

    def __init__(self, threshold: float = 7.0, judge_model: str = "openai:gpt-4o-mini"):
        """
        Initialize quality evaluator.

        Args:
            threshold: Minimum quality score (0-10)
            judge_model: Model to use for evaluation
        """
        self.threshold = threshold
        self.judge_model = judge_model
        self._judge_agent = Agent(judge_model)

    async def evaluate(self, prompt: str, result: Any, context: dict) -> None:
        """
        Evaluate response quality using LLM-as-judge.

        Args:
            prompt: User prompt
            result: Agent result
            context: Additional context
        """
        try:
            # Build evaluation prompt
            eval_prompt = f"""Rate the quality of this AI response on a scale of 0-10.
Consider: relevance, accuracy, helpfulness, clarity, and completeness.

User Prompt: {prompt}

AI Response: {result.output if hasattr(result, "output") else str(result)}

Respond with just a number between 0 and 10."""

            # Get judgment
            judgment = await self._judge_agent.run(eval_prompt)

            try:
                score = float(judgment.output.strip())
            except ValueError:
                logger.warning(
                    "Could not parse quality score", judgment=judgment.output
                )
                return

            # Log result
            if score < self.threshold:
                logger.warning(
                    "Low quality response detected",
                    score=score,
                    threshold=self.threshold,
                    prompt=prompt[:100],
                    **context,
                )
            else:
                logger.info(
                    "Quality check passed",
                    score=score,
                    threshold=self.threshold,
                    **context,
                )

        except Exception as e:
            logger.error(f"Quality evaluation failed: {str(e)}")


class SafetyCheck:
    """Content safety evaluation using OpenAI moderation API."""

    def __init__(self):
        """Initialize safety evaluator."""
        pass

    async def evaluate(self, prompt: str, result: Any, context: dict) -> None:
        """
        Check content safety using OpenAI moderation API.

        Args:
            prompt: User prompt
            result: Agent result
            context: Additional context
        """
        try:
            import openai

            # Get result text
            result_text = result.output if hasattr(result, "output") else str(result)

            # Check both prompt and result
            moderation = await openai.moderations.create(input=[prompt, result_text])

            # Check if any content was flagged
            for i, mod_result in enumerate(moderation.results):
                content_type = "prompt" if i == 0 else "response"

                if mod_result.flagged:
                    categories = [
                        cat
                        for cat, flagged in mod_result.categories.model_dump().items()
                        if flagged
                    ]

                    logger.warning(
                        f"Content policy violation in {content_type}",
                        categories=categories,
                        content=content_type,
                        **context,
                    )
                else:
                    logger.debug(f"Safety check passed for {content_type}", **context)

        except ImportError:
            logger.warning("OpenAI not available - skipping safety check")
        except Exception as e:
            logger.error(f"Safety evaluation failed: {str(e)}")


class CustomEvaluator:
    """
    Base class for custom evaluators.

    Example usage:
        class MyEvaluator(CustomEvaluator):
            async def evaluate(self, prompt, result, context):
                # Custom evaluation logic
                if some_condition:
                    self.log_warning("Issue detected", details="...")
    """

    def __init__(self, name: str = "custom"):
        """
        Initialize custom evaluator.

        Args:
            name: Evaluator name for logging
        """
        self.name = name
        self.logger = structlog.get_logger()

    def log_info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(f"[{self.name}] {message}", **kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(f"[{self.name}] {message}", **kwargs)

    def log_error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(f"[{self.name}] {message}", **kwargs)

    async def evaluate(self, prompt: str, result: Any, context: dict) -> None:
        """
        Override this method with custom evaluation logic.

        Args:
            prompt: User prompt
            result: Agent result
            context: Additional context
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
