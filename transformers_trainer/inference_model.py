import time

import librosa
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

from utils import logger, parse_inference_args


def inference(
    model_path: str,
    input_audio_path: str,
    precision: str,
) -> None:
    """
    Performs speech-to-text transcription using a Whisper model.

    Args:
        model_path (str): Path to the pretrained Whisper model or a Hugging Face model ID.
        input_audio_path (str): Path to the input audio file for transcription.
        precision (str): Floating point precision for inference. Supported values: "fp32", "fp16", "bf16".
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if precision == "fp32":
        torch_dtype = torch.float32
    elif precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16

    logger.info(msg=f"Model path: {model_path}")
    logger.info(msg=f"Input audio path: {input_audio_path}")
    logger.info(msg=f"Device: {device}")
    logger.info(msg=f"Torch dtype: {torch_dtype}")

    start = time.time()
    tokenizer = WhisperTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_path,
        task="transcribe",
    )
    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path,
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = True
    logger.info(msg=f"Model loaded in {time.time() - start:.3f} seconds.")

    start = time.time()
    audio, _ = librosa.load(
        path=input_audio_path,
        sr=16000,
        mono=True,
    )
    logger.info(msg=f"Audio loaded in {time.time() - start:.3f} seconds.")

    start = time.time()
    input_features = processor(
        audio=audio,
        return_tensors="pt",
        sampling_rate=16000,
    ).input_features.to(
        device=device,
        dtype=torch_dtype,
    )
    predicted_ids = model.generate(input_features=input_features)
    transcription = tokenizer.batch_decode(
        sequences=predicted_ids,
        skip_special_tokens=True,
    )[0]
    logger.info(msg=f"Transcription generated in {time.time() - start:.3f} seconds.")

    logger.info(msg=f"Transcription: {transcription}")
    logger.info(msg=f"\n{torch.cuda.memory_summary(device=device)}")


if __name__ == "__main__":
    args = parse_inference_args()
    inference(
        model_path=args.model_path,
        input_audio_path=args.input_audio_path,
        precision=args.precision,
    )
