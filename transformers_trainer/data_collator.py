from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    A data collator for speech-to-text sequence-to-sequence models that handles
    padding for both input features and labels.

    Attributes:
        tokenizer (WhisperTokenizerFast): The tokenizer used for processing text data.
        feature_extractor (WhisperFeatureExtractor): The feature extractor for audio processing.
        model_decoder_start_token_id (int): The token ID that marks the start of decoding.
    """

    tokenizer: WhisperTokenizerFast
    feature_extractor: WhisperFeatureExtractor
    model_decoder_start_token_id: int

    def __call__(
        cls,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Pads input features and labels to create a batch for speech-to-text training.

        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]):
                A list of feature dictionaries containing "input_features" and "labels".

        Returns:
            Dict[str, torch.Tensor]:
                A dictionary containing padded input features and labels,
                formatted for model training.
        """
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = cls.feature_extractor.pad(
            processed_features=input_features,
            return_tensors="pt",
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = cls.tokenizer.pad(
            encoded_inputs=label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )
        if (labels[:, 0] == cls.model_decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
