import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

# Configure logging to display timestamps, module name, log level, and message
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Create a logger for tracking execution
logger = logging.getLogger(name="CTranslate2-Engine")


def parse_inference_args() -> Namespace:
    """
    Parses command-line arguments for performing inference with a Whisper model.

    Returns:
        Namespace: Parsed arguments containing the model path and input audio path.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path or Hugging Face model ID for the model to be used for inference.",
    )
    parser.add_argument(
        "-i",
        "--input_audio_path",
        type=Path,
        required=True,
        help="Path to the input audio file for inference.",
    )
    return parser.parse_args()


def parse_push_args() -> Namespace:
    """
    Parses command-line arguments for pushing a trained model to the Hugging Face Hub.

    Returns:
        Namespace: Parsed arguments containing model path, repository ID, commit message, and privacy flag.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to be pushed to the Hugging Face Hub.",
    )
    parser.add_argument(
        "-id",
        "--huggingface_repo_id",
        type=str,
        required=True,
        help="Hub repository ID for the model to be pushed.",
    )
    parser.add_argument(
        "-cm",
        "--commit_message",
        type=str,
        required=True,
        help="Commit message for the model to be pushed.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Flag to indicate if the model should be private on the Hugging Face Hub.",
    )
    return parser.parse_args()
