from dataclasses import dataclass
from typing import Any, Dict

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

from configs import TrainConfigs
from system import AUDIO_SAMPLE_RATE, DATASET_WRITER_BATCH_SIZE, NUM_WORKERS


@dataclass
class DataLoader:
    """
    A class responsible for preparing and processing datasets for training and evaluation.

    Attributes:
        feature_extractor (WhisperFeatureExtractor): The feature extractor for processing audio data.
        tokenizer (WhisperTokenizerFast): The tokenizer for converting text transcriptions into tokenized inputs.
        model_max_length (int): The maximum length of tokenized sequences.
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizerFast
    model_max_length: int

    def prepare_dataset(
        cls,
        configs: TrainConfigs,
    ) -> DatasetDict:
        """
        Loads, processes, and prepares datasets for training and testing.

        Args:
            configs (TrainConfigs): The training configuration containing dataset information.

        Returns:
            DatasetDict: A dictionary containing processed "train" and "test" datasets.
        """
        dataset = DatasetDict()
        train_dataset_list = []
        test_dataset_list = []
        for d in configs.datasets.values():
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
        Preprocesses a batch of data by extracting audio features and tokenizing transcriptions.

        Args:
            batch (Dict[str, Any]): A dictionary containing an audio sample and its transcription.

        Returns:
            Dict[str, Any]: A dictionary with processed input features and tokenized labels.
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
