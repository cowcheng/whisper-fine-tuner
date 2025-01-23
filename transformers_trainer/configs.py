from typing import Dict

from pydantic import BaseModel, Field


class SubsetConfigs(BaseModel):
    """
    Configuration settings for a specific subset of the dataset.

    Attributes:
        name (str): The name identifier for the subset.
        split (str): The data split type, such as 'train', 'validation', or 'test'.
        num_rows (int): The number of rows or samples in the subset.
    """

    name: str = Field()
    split: str = Field()
    num_rows: int = Field()


class DataConfigs(BaseModel):
    """
    Configuration settings for the dataset used in training and evaluation.

    Attributes:
        dataset_name (str): The name of the dataset.
        subsets (Dict[str, SubsetConfigs]): A dictionary of subset configurations, keyed by subset name.
        audio_column (str): The name of the column containing audio data.
        transcription_column (str): The name of the column containing transcription text.
        test_size (float): The proportion of the dataset to include in the test split.
    """

    dataset_name: str = Field()
    subsets: Dict[str, SubsetConfigs] = Field()
    audio_column: str = Field()
    transcription_column: str = Field()
    test_size: float = Field()


class TrainerConfigs(BaseModel):
    """
    Configuration settings for the training process.

    Attributes:
        pretrained_model_name (str): The name of the pretrained model to use.
        per_device_train_batch_size (int): Batch size per device during training.
        per_device_eval_batch_size (int): Batch size per device during evaluation.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating.
        learning_rate (float): The initial learning rate for the optimizer.
        num_train_epochs (int): Total number of training epochs.
        lr_scheduler_type (str): The type of learning rate scheduler to use.
        warmup_steps (int): Number of steps for the warmup phase.
        logging_steps (int): Interval (in steps) at which to log training metrics.
        save_steps (int): Interval (in steps) at which to save model checkpoints.
        precision (str): The numerical precision to use (e.g., 'fp16', 'bf16').
        eval_steps (int): Interval (in steps) at which to perform evaluation.
        gradient_checkpointing (bool): Whether to enable gradient checkpointing to save memory.
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


class TransformersTrainerConfigs(BaseModel):
    """
    Configuration settings combining data and trainer configurations for the Transformers trainer.

    Attributes:
        data (Dict[str, DataConfigs]): A dictionary of data configurations, keyed by dataset name.
        trainer (TrainerConfigs): The trainer configuration settings.
    """

    data: Dict[str, DataConfigs] = Field()
    trainer: TrainerConfigs = Field()
