from typing import Any, Dict

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from configs import TrainConfigs
from system import (
    AUDIO_SAMPLE_RATE,
    DATASET_WRITER_BATCH_SIZE,
    NUM_WORKERS,
)
from utils import logger, parse_args, read_yaml

# -------------------------------
# Load configurations
# -------------------------------
args = parse_args()
configs_dict = read_yaml(yaml_path=args.configs_path)
configs = TrainConfigs(**configs_dict)
logger.info(msg=f"Configs: {configs}")

# -------------------------------
# Initialize model components
# -------------------------------
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path=configs.trainer.pretrained_model_name,
)
tokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model_name_or_path=configs.trainer.pretrained_model_name,
    task="transcribe",
)
model = WhisperForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=configs.trainer.pretrained_model_name,
    device_map="cpu",
)


# -------------------------------
# Define dataset preprocessing function
# -------------------------------
def preprocess_dataset(
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        raw_speech=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        device="cuda:0",
    ).input_features[0]

    batch["labels"] = tokenizer(text=batch["transcription"]).input_ids

    return batch


# -------------------------------
# Load and process datasets
# -------------------------------
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

# -------------------------------
# Filter and preprocess dataset
# -------------------------------
dataset = dataset.filter(
    function=lambda x: len(x["transcription"]) < model.config.max_length,
    cache_file_names={k: f"caches/filter/{str(k)}.arrow" for k in dataset},
    writer_batch_size=DATASET_WRITER_BATCH_SIZE,
    num_proc=NUM_WORKERS,
)
dataset = dataset.cast_column(
    column="audio",
    feature=Audio(sampling_rate=AUDIO_SAMPLE_RATE),
)
dataset = dataset.map(
    function=preprocess_dataset,
    remove_columns=dataset.column_names["train"],
    cache_file_names={k: f"caches/map/{str(k)}.arrow" for k in dataset},
    writer_batch_size=DATASET_WRITER_BATCH_SIZE,
    num_proc=NUM_WORKERS,
)
