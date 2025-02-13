from typing import Dict

from pydantic import BaseModel, Field


class DatasetSubsetConfigs(BaseModel):
    """
    Configuration for a specific subset of a dataset.

    Attributes:
        name (str): The name of the subset.
        split (str): The dataset split (e.g., "train", "test", "validation").
        num_rows (int): The number of rows in the subset.
    """

    name: str = Field()
    split: str = Field()
    num_rows: int = Field()


class TrainDatasetsConfigs(BaseModel):
    """
    Configuration for training datasets.

    Attributes:
        dataset_name (str): The name of the dataset.
        subsets (Dict[str, DatasetSubsetConfigs]): A dictionary of dataset subsets.
        audio_column (str): The column name for audio data.
        transcription_column (str): The column name for transcription data.
        test_size (float): The proportion of the dataset to use for testing.
    """

    dataset_name: str = Field()
    subsets: Dict[str, DatasetSubsetConfigs] = Field()
    audio_column: str = Field()
    transcription_column: str = Field()
    test_size: float = Field()


class TrainerConfigs(BaseModel):
    """
    Configuration for the training process.

    Attributes:
        pretrained_model_name (str): Name of the pretrained model to use.
        per_device_train_batch_size (int): Training batch size per device.
        per_device_eval_batch_size (int): Evaluation batch size per device.
        gradient_accumulation_steps (int): Number of steps for gradient accumulation.
        learning_rate (float): The learning rate for training.
        num_train_epochs (int): Total number of training epochs.
        lr_scheduler_type (str): The type of learning rate scheduler.
        warmup_steps (int): Number of warmup steps before reaching full learning rate.
        logging_steps (int): Frequency of logging steps.
        save_steps (int): Frequency of model saving steps.
        precision (str): The precision type (e.g., "fp16", "bf16").
        eval_steps (int): Frequency of evaluation steps.
        gradient_checkpointing (bool): Whether to use gradient checkpointing to save memory.
    """

    pretrained_model_name: str = Field()
    per_device_train_batch_size: int = Field()
    per_device_eval_batch_size: int = Field()
    gradient_accumulation_steps: int = Field()
    learning_rate: float = Field()
    num_train_epochs: int = Field()
    lr_scheduler_type: str = Field()
    warmup_steps: int = Field()
    logging_steps: int = Field()
    save_steps: int = Field()
    precision: str = Field()
    eval_steps: int = Field()
    gradient_checkpointing: bool = Field()


class TrainConfigs(BaseModel):
    """
    Configuration for the full training pipeline.

    Attributes:
        datasets (Dict[str, TrainDatasetsConfigs]): A dictionary of dataset configurations.
        trainer (TrainerConfigs): The trainer configurations.
    """

    datasets: Dict[str, TrainDatasetsConfigs] = Field()
    trainer: TrainerConfigs = Field()


class EvaluatorConfigs(BaseModel):
    """
    Configuration for the evaluation process.

    Attributes:
        model_path (str): The path to the trained model for evaluation.
        precision (str): Floating point precision for inference. Supported values: "fp32", "fp16", "bf16".
    """

    model_path: str = Field()
    precision: str = Field()


class EvalDatasetsConfigs(BaseModel):
    """
    Configuration for evaluation datasets.

    Attributes:
        dataset_name (str): The name of the dataset.
        subsets (Dict[str, DatasetSubsetConfigs]): A dictionary of dataset subsets.
        audio_column (str): The column name for audio data.
        transcription_column (str): The column name for transcription data.
    """

    dataset_name: str = Field()
    subsets: Dict[str, DatasetSubsetConfigs] = Field()
    audio_column: str = Field()
    transcription_column: str = Field()


class EvalConfigs(BaseModel):
    """
    Configuration for the full evaluation pipeline.

    Attributes:
        datasets (Dict[str, EvalDatasetsConfigs]): A dictionary of dataset configurations.
        evaluator (EvaluatorConfigs): The evaluator configurations.
    """

    datasets: Dict[str, EvalDatasetsConfigs] = Field()
    evaluator: EvaluatorConfigs = Field()
