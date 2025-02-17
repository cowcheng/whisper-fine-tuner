from pathlib import Path

from huggingface_hub import HfApi

from utils import parse_push_args

# Initialize Hugging Face API client
api = HfApi()


def upload(
    model_path: Path,
    huggingface_repo_id: str,
    commit_message: str,
    private: bool,
) -> None:
    """
    Uploads a model to the Hugging Face Hub.

    Args:
        model_path (Path): Path to the local model directory to be uploaded.
        huggingface_repo_id (str): Repository ID on the Hugging Face Hub.
        commit_message (str): Commit message describing the upload.
        private (bool): Whether the repository should be private.
    """
    api.create_repo(
        repo_id=huggingface_repo_id,
        private=private,
        repo_type="model",
    )

    api.upload_folder(
        repo_id=huggingface_repo_id,
        folder_path=model_path,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    args = parse_push_args()
    upload(
        model_path=args.model_path,
        huggingface_repo_id=args.huggingface_repo_id,
        commit_message=args.commit_message,
        private=args.private,
    )
