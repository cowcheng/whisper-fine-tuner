import time
from pathlib import Path

from faster_whisper import WhisperModel

from utils import logger, parse_inference_args


def inference(
    model_path: str,
    input_audio_path: Path,
) -> None:
    """
    Runs inference using a Whisper model on the provided audio file.

    Args:
        model_path (str): Path to the Whisper model or Hugging Face model ID.
        input_audio_path (Path): Path to the input audio file for transcription.
    """
    logger.info(msg=f"Model path: {model_path}")
    logger.info(msg=f"Input audio path: {input_audio_path}")

    start = time.time()
    ct2_model = WhisperModel(model_size_or_path=model_path)
    logger.info(msg=f"Model loaded in {time.time() - start:.3f} seconds.")

    start = time.time()
    segments, info = ct2_model.transcribe(audio=input_audio_path)
    logger.info(msg=f"Inference completed in {time.time() - start:.3f} seconds.")

    logger.info(msg=f"Info: {info}")
    for segment in segments:
        logger.info(msg=f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")


if __name__ == "__main__":
    args = parse_inference_args()
    inference(
        model_path=args.model_path,
        input_audio_path=args.input_audio_path,
    )
