import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict

import yaml

# Configure the logging settings
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Create a logger instance for this module
logger = logging.getLogger(name="Transformers-Trainer")


def parse_args() -> Namespace:
    """
    Parse command-line arguments provided by the user.

    This function sets up an argument parser to handle command-line inputs.
    Currently, it requires the user to provide the path to a YAML configuration file.

    Returns:
        Namespace: An object containing the parsed command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--configs_path",
        type=Path,
        required=True,
        help="Path to the YAML configuration file for fine-tuning the Whisper model",
    )
    return parser.parse_args()


def read_yaml(
    yaml_path: Path,
) -> Dict[str, Any]:
    """
    Read and parse a YAML configuration file.

    This function opens the specified YAML file, parses its content, and returns
    the configurations as a dictionary.

    Args:
        yaml_path (Path): The file system path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configurations from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist at the specified path.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(yaml_path) as fs:
        configs = yaml.safe_load(stream=fs)
    return configs
