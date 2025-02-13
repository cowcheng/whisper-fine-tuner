from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils import logger, parse_push_args


def upload(
    model_path: str,
    huggingface_repo_id: str,
    commit_message: str,
    private: bool = False,
) -> None:
    """
    Uploads a Whisper model and its processor to the Hugging Face Hub.

    Args:
        model_path (str): The local path to the trained Whisper model.
        huggingface_repo_id (str): The repository ID on Hugging Face where the model will be uploaded.
        commit_message (str): The commit message for the upload.
        private (bool, optional): Whether to make the repository private. Defaults to False.
    """
    logger.info(msg=f"Model path: {model_path}")
    logger.info(msg=f"Hugging Face repo ID: {huggingface_repo_id}")
    logger.info(msg=f"Commit message: {commit_message}")
    logger.info(msg=f"Private: {private}")

    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path,
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )

    processor.push_to_hub(
        repo_id=huggingface_repo_id,
        commit_message=commit_message,
        private=private,
    )

    model.push_to_hub(
        repo_id=huggingface_repo_id,
        commit_message=commit_message,
        private=private,
    )


if __name__ == "__main__":
    args = parse_push_args()
    upload(
        model_path=args.model_path,
        huggingface_repo_id=args.huggingface_repo_id,
        commit_message=args.commit_message,
        private=args.private,
    )
