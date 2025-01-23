from dataclasses import dataclass
from typing import Any, Dict

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

from configs import TransformersTrainerConfigs
from system import AUDIO_SAMPLE_RATE, DATASET_WRITER_BATCH_SIZE, NUM_WORKERS


@dataclass
class DataLoader:
    """
    DataLoader is responsible for preparing and preprocessing datasets for training
    and evaluation using the Whisper model. It handles loading datasets, renaming and
    selecting necessary columns, splitting into training and testing sets, filtering,
    casting audio columns, and mapping preprocessing functions to the dataset.

    Attributes:
        feature_extractor (WhisperFeatureExtractor): Extracts features from raw audio data.
        tokenizer (WhisperTokenizerFast): Tokenizes transcription text for the model.
        model_max_length (int): Maximum allowed length for model inputs.
        model_sampling_rate (int, optional): Sampling rate for audio data. Defaults to 16000.
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizerFast
    model_max_length: int
    model_sampling_rate: int = 16000

    def prepare_dataset(
        cls,
        configs: TransformersTrainerConfigs,
    ) -> DatasetDict:
        """
        Prepares the dataset for training and testing by loading, processing, and
        concatenating multiple dataset subsets as specified in the configuration.

        This method performs the following steps:
            1. Loads each dataset subset based on the provided configurations.
            2. Renames relevant columns to standardized names ("audio" and "transcription").
            3. Selects only the necessary columns.
            4. Splits each subset into training and testing partitions.
            5. Concatenates all training and testing subsets into unified datasets.
            6. Filters out samples where the transcription exceeds the maximum model length.
            7. Casts the audio column to the appropriate audio feature with the specified sampling rate.
            8. Applies preprocessing to each dataset split.

        Args:
            configs (TransformersTrainerConfigs): Configuration object containing dataset details.

        Returns:
            DatasetDict: A dictionary containing the prepared training and testing datasets.
        """
        dataset = DatasetDict()
        train_dataset_list = []
        test_dataset_list = []
        for d in configs.data.values():
            for s in d.subsets.values():
                tmp_dataset = load_dataset(
                    path=d.dataset_name,
                    name=s.name,
                    split=s.split,
                    num_proc=NUM_WORKERS,
                ).select(
                    indices=range(s.num_rows),
                    writer_batch_size=DATASET_WRITER_BATCH_SIZE,
                )
                tmp_dataset = tmp_dataset.rename_columns(
                    column_mapping={
                        d.audio_column: "audio",
                        d.transcription_column: "transcription",
                    }
                )
                tmp_dataset = tmp_dataset.select_columns(
                    column_names=[
                        "audio",
                        "transcription",
                    ]
                )
                tmp_dataset = tmp_dataset.train_test_split(
                    test_size=d.test_size,
                    writer_batch_size=DATASET_WRITER_BATCH_SIZE,
                )
                train_dataset_list.append(tmp_dataset["train"])
                test_dataset_list.append(tmp_dataset["test"])
        dataset["train"] = concatenate_datasets(dsets=train_dataset_list)
        dataset["test"] = concatenate_datasets(dsets=test_dataset_list)
        dataset = dataset.filter(
            function=lambda x: len(x["transcription"]) < cls.model_max_length,
            cache_file_names={k: f"caches/filter/{str(k)}.arrow" for k in dataset},
            writer_batch_size=DATASET_WRITER_BATCH_SIZE,
            num_proc=NUM_WORKERS,
        )
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(sampling_rate=AUDIO_SAMPLE_RATE),
        )
        dataset = dataset.map(
            function=cls.preprocess_dataset,
            remove_columns=dataset.column_names["train"],
            cache_file_names={k: f"caches/map/{str(k)}.arrow" for k in dataset},
            writer_batch_size=DATASET_WRITER_BATCH_SIZE,
            num_proc=NUM_WORKERS,
        )

        return dataset

    def preprocess_dataset(
        cls,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Preprocesses a batch of dataset samples by extracting input features from audio
        and tokenizing the transcription text.

        This method performs the following steps:
            1. Extracts raw audio data and sampling rate from the batch.
            2. Uses the feature extractor to convert raw audio into model-compatible input features.
            3. Tokenizes the transcription text into input IDs for the model.

        Args:
            batch (Dict[str, Any]): A dictionary containing a batch of dataset samples with
                'audio' and 'transcription' keys.

        Returns:
            Dict[str, Any]: A dictionary with added 'input_features' and 'labels' keys for
                model training.
        """
        audio = batch["audio"]

        batch["input_features"] = cls.feature_extractor(
            raw_speech=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            device="cuda:0",
        ).input_features[0]

        batch["labels"] = cls.tokenizer(text=batch["transcription"]).input_ids

        return batch
