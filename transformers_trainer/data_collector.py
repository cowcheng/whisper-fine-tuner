from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the inputs and labels for speech sequence-to-sequence models.

    This collator is designed to work with Whisper models from the Hugging Face Transformers library.
    It handles padding of input features (e.g., audio features) and tokenized labels (e.g., transcriptions),
    ensuring that all inputs in a batch are of equal length. It also prepares the labels by masking
    non-relevant tokens, which is essential for training sequence-to-sequence models.

    Attributes:
        tokenizer (WhisperTokenizerFast): The tokenizer used to process the target text sequences.
        feature_extractor (WhisperFeatureExtractor): The feature extractor used to process audio inputs.
        model_decoder_start_token_id (int): The token ID that indicates the start of decoding in the model.
    """

    tokenizer: WhisperTokenizerFast
    feature_extractor: WhisperFeatureExtractor
    model_decoder_start_token_id: int

    def __call__(
        cls,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of features by padding input features and labels to create a uniform batch.

        This method performs the following steps:
        1. Extracts and pads the input audio features using the feature extractor.
        2. Extracts and pads the target labels (token IDs) using the tokenizer.
        3. Masks the padding tokens in the labels to ignore them during loss computation.
        4. Removes the initial start token from the labels if present.

        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]):
                A list of feature dictionaries, each containing:
                    - "input_features": A tensor or list of integers representing the input audio features.
                    - "labels": A list of integers representing the tokenized target sequence.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "input_features": A padded tensor of input audio features.
                - "labels": A padded tensor of target token IDs with appropriate masking.
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
