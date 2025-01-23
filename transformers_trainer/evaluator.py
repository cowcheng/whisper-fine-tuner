from dataclasses import dataclass
from typing import Any, Dict

import evaluate
from evaluate import EvaluationModule
from transformers import WhisperTokenizerFast

# Load the Word Error Rate (WER) metric from the `evaluate` library
metric: EvaluationModule = evaluate.load(path="wer")


@dataclass
class Evaluator:
    """
    Evaluator for computing metrics on speech recognition model predictions.

    This class utilizes the WhisperTokenizerFast to decode predicted and label token IDs
    into strings and computes the Word Error Rate (WER) between them using the `evaluate` library.

    Attributes:
        tokenizer (WhisperTokenizerFast): The tokenizer used to decode token IDs into text.
    """

    tokenizer: WhisperTokenizerFast

    def compute_metrics(
        cls,
        pred: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Computes evaluation metrics for model predictions.

        Specifically, this method decodes the predicted and label token IDs into strings,
        replaces padding tokens in labels, and calculates the Word Error Rate (WER) between
        the predictions and references.

        Args:
            pred (Dict[str, Any]):
                A dictionary containing:
                    - "predictions": A tensor or array of predicted token IDs from the model.
                    - "label_ids": A tensor or array of true label token IDs.

        Returns:
            Dict[str, float]: A dictionary with the computed metrics. Currently, it includes:
                - "wer": The Word Error Rate as a percentage.
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

        wer = 100 * metric.compute(
            predictions=pred_str,
            references=label_str,
        )

        return {"wer": wer}
