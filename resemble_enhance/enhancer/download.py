import logging
import os
import subprocess
from pathlib import Path

REPO_URL = "https://huggingface.co/ResembleAI/resemble-enhance"
REPO_DIR = Path(__file__).parent.parent / "model_repo"

logger = logging.getLogger(__name__)


def run_command(command, msg=None, env={}):
    try:
        subprocess.run(command, check=True, env={**os.environ, **env})
    except subprocess.CalledProcessError as e:
        if msg is not None:
            raise RuntimeError(msg) from e
        raise e


def download():
    logger.info("Downloading the model...")

    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        logger.info("Repository already exists, attempting to pull latest changes...")
        run_command(
            ["git", "-C", str(REPO_DIR), "pull"],
            "Failed to pull latest changes, please try again.",
            {"GIT_LFS_SKIP_SMUDGE": "1"},
        )
    else:
        logger.info("Cloning the repository...")
        run_command(
            ["git", "clone", REPO_URL, str(REPO_DIR)],
            "Failed to clone the repository, please try again.",
            {"GIT_LFS_SKIP_SMUDGE": "1"},
        )

    logger.info("Pulling large files...")
    run_command(["git", "-C", str(REPO_DIR), "lfs", "pull"], "Failed to pull latest changes, please try again.")

    run_dir = REPO_DIR / "enhancer_stage2"

    return run_dir
