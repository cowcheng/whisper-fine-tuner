from dataclasses import dataclass
from typing import Any, Dict

import evaluate
from evaluate import EvaluationModule
from transformers import WhisperTokenizerFast

metric: EvaluationModule = evaluate.load(path="cer")


@dataclass
class Evaluator:
    """
    A class for evaluating speech-to-text model performance using Character Error Rate (CER).

    Attributes:
        tokenizer (WhisperTokenizerFast): The tokenizer used for decoding predictions and labels.
    """

    tokenizer: WhisperTokenizerFast

    def compute_metrics(
        cls,
        pred: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Computes the Character Error Rate (CER) for model predictions.

        Args:
            pred (Dict[str, Any]): A dictionary containing:
                - `predictions`: The predicted token IDs.
                - `label_ids`: The actual token IDs, where `-100` values indicate ignored tokens.

        Returns:
            Dict[str, float]: A dictionary containing the computed CER as a percentage.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = cls.tokenizer.pad_token_id

        pred_str = cls.tokenizer.batch_decode(
            sequences=pred_ids,
            skip_special_tokens=True,
        )
        label_str = cls.tokenizer.batch_decode(
            sequences=label_ids,
            skip_special_tokens=True,
        )

        cer = 100 * metric.compute(
            predictions=pred_str,
            references=label_str,
        )

        return {"cer": cer}
