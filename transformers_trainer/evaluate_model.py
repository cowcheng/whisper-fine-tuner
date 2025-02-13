import evaluate
from datasets import concatenate_datasets, load_dataset
from evaluate import EvaluationModule
import torch
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

from configs import EvalConfigs
from system import NUM_WORKERS
from utils import logger, parse_args, read_yaml

metric: EvaluationModule = evaluate.load(path="cer")


def eval(
    configs: EvalConfigs,
) -> None:
    """
    Evaluates a Whisper model on a given dataset by computing the Character Error Rate (CER).

    Args:
        configs (EvalConfigs): Evaluation configurations containing dataset paths, model path, and evaluation settings.
    """
    logger.info(msg=f"Configs: {configs.__dict__}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if configs.evaluator.precision == "fp32":
        torch_dtype = torch.float32
    elif configs.evaluator.precision == "fp16":
        torch_dtype = torch.float16
    elif configs.evaluator.precision == "bf16":
        torch_dtype = torch.bfloat16

    logger.info(msg=f"Device: {device}")
    logger.info(msg=f"Torch dtype: {torch_dtype}")

    # Load and process evaluation datasets
    eval_datasets_list = []
    for d in configs.datasets.values():
        for s in d.subsets.values():
            tmp_dataset = load_dataset(
                path=d.dataset_name,
                name=s.name,
                split=s.split,
                num_proc=NUM_WORKERS,
            )
            if s.num_rows != -1:
                tmp_dataset = tmp_dataset.select(indices=range(s.num_rows))
            eval_datasets_list.append(tmp_dataset)
    eval_dataset = concatenate_datasets(dsets=eval_datasets_list)

    # Load the Whisper processor, tokenizer, and model
    tokenizer = WhisperTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=configs.evaluator.model_path,
        task="transcribe",
    )
    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=configs.evaluator.model_path,
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=configs.evaluator.model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = True

    # Iterate through evaluation dataset and generate predictions
    gt_text_list = []
    pred_text_list = []
    for data in tqdm(
        iterable=eval_dataset,
        total=len(eval_dataset),
    ):
        for d in configs.datasets.values():
            audio_column = d.audio_column
            transcription_column = d.transcription_column
            audio = data[audio_column]["array"]
            transcription = data[transcription_column]
            input_features = processor(
                audio=audio,
                return_tensors="pt",
                sampling_rate=16000,
            ).input_features.to(
                device=device,
                dtype=torch_dtype,
            )
            predicted_ids = model.generate(input_features=input_features)
            prediction = tokenizer.batch_decode(
                sequences=predicted_ids,
                skip_special_tokens=True,
            )[0]
            gt_text_list.append(transcription)
            pred_text_list.append(prediction)

    # Compute Character Error Rate (CER)
    cer = metric.compute(
        predictions=pred_text_list,
        references=gt_text_list,
    )
    logger.info(msg=f"CER: {cer * 100}")


if __name__ == "__main__":
    args = parse_args()
    configs_dict = read_yaml(yaml_path=args.configs_path)
    configs = EvalConfigs(**configs_dict)
    eval(configs=configs)
