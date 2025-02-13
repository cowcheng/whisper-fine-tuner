import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict

import yaml

# Configure logging to display timestamps, module name, log level, and message
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Create a logger for tracking execution
logger = logging.getLogger(name="Transformers-Trainer")


def parse_args() -> Namespace:
    """
    Parses command-line arguments for specifying a YAML configuration file.

    Returns:
        Namespace: Parsed arguments containing the path to the YAML configuration file.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--configs_path",
        type=Path,
        required=True,
        help="Path to the YAML configuration file for fine-tuning or evaluate the Whisper model.",
    )
    return parser.parse_args()


def parse_inference_args() -> Namespace:
    """
    Parses command-line arguments for performing inference with a Whisper model.

    Returns:
        Namespace: Parsed arguments containing the model path, input audio path, and precision.
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
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision to use for inference (default: fp32).",
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


def read_yaml(
    yaml_path: Path,
) -> Dict[str, Any]:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        yaml_path (Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed contents of the YAML file.
    """
    with open(yaml_path) as fs:
        configs = yaml.safe_load(stream=fs)
    return configs
