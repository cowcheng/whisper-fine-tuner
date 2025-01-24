import os

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

from configs import TransformersTrainerConfigs
from data_collector import DataCollatorSpeechSeq2SeqWithPadding
from data_loader import DataLoader
from evaluator import Evaluator
from system import NUM_WORKERS
from utils import logger, parse_args, read_yaml

# TODO: Fix later
# Disable the warning messages from the Transformers library
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# Enable TensorFloat-32 for matrix multiplications on CUDA
torch.backends.cuda.matmul.allow_tf32 = True


def train(
    configs: TransformersTrainerConfigs,
) -> None:
    """
    Trains the Whisper model using the provided configurations.

    This function sets up the training environment, including training arguments,
    model components, datasets, data collator, evaluator, and the trainer itself.
    It then initiates the training process and logs relevant information.

    Args:
        configs (TransformersTrainerConfigs): Configuration object containing
            all necessary parameters for training, such as dataset paths,
            model settings, training hyperparameters, and more.

    Raises:
        ValueError: If an unsupported precision type is specified in the configurations.
    """
    # Initialize training arguments for the Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models",
        eval_strategy="steps",
        per_device_train_batch_size=configs.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=configs.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=configs.trainer.gradient_accumulation_steps,
        learning_rate=configs.trainer.learning_rate,
        num_train_epochs=configs.trainer.num_train_epochs,
        lr_scheduler_type=configs.trainer.lr_scheduler_type,
        warmup_steps=configs.trainer.warmup_steps,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=configs.trainer.logging_steps,
        save_strategy="steps",
        save_steps=configs.trainer.save_steps,
        bf16=True if configs.trainer.precision == "bf16" else False,
        fp16=True if configs.trainer.precision == "fp16" else False,
        bf16_full_eval=True if configs.trainer.precision == "bf16" else False,
        fp16_full_eval=True if configs.trainer.precision == "fp16" else False,
        ddp_backend="nccl" if torch.cuda.device_count() > 1 else None,
        eval_steps=configs.trainer.eval_steps,
        dataloader_num_workers=NUM_WORKERS,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=["tensorboard"],
        gradient_checkpointing=configs.trainer.gradient_checkpointing,
        eval_on_start=True,
        predict_with_generate=True,
    )

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        logger.info(msg=f"Configs: {configs.__dict__}")
        logger.info(msg=f"Training arguments: {training_args.__dict__}")

    # Determine the GPU ID based on the local rank
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": gpu_id}

    # Load feature extractor, tokenizer, and processor from the pretrained model
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=configs.trainer.pretrained_model_name,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=configs.trainer.pretrained_model_name,
        task="transcribe",
    )
    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=configs.trainer.pretrained_model_name,
        task="transcribe",
    )

    # Load the Whisper model for conditional generation with the specified device map
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=configs.trainer.pretrained_model_name,
        device_map=device_map,
    )

    # Configure the model settings
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Initialize the DataLoader with the feature extractor and tokenizer
    data_loader = DataLoader(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        model_max_length=model.config.max_length,
    )

    # Prepare the dataset using the DataLoader and provided configurations
    dataset = data_loader.prepare_dataset(configs=configs)
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        logger.info(msg=f"Dataset: {dataset}")

    # Initialize the data collator with padding for speech-to-seq2seq tasks
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model_decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Initialize the evaluator with the tokenizer for computing metrics
    evaluator = Evaluator(tokenizer=tokenizer)

    # Initialize the Seq2SeqTrainer with all necessary components
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=feature_extractor,
        compute_metrics=evaluator.compute_metrics,
    )

    # Save the processor (feature extractor and tokenizer) for future use
    processor.save_pretrained(save_directory="./models/processor")

    # Start the training process
    trainer.train()

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        logger.info(msg="========== Finetuning complete ==========")


if __name__ == "__main__":
    args = parse_args()
    configs_dict = read_yaml(yaml_path=args.configs_path)
    configs = TransformersTrainerConfigs(**configs_dict)
    train(configs=configs)
