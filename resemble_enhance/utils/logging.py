import logging
from pathlib import Path

from rich.logging import RichHandler

from .distributed import global_leader_only


@global_leader_only
def setup_logging(run_dir):
    handlers = []
    stdout_handler = RichHandler()
    stdout_handler.setLevel(logging.INFO)
    handlers.append(stdout_handler)

    if run_dir is not None:
        filename = Path(run_dir) / f"log.txt"
        filename.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    # Update all existing loggers
    for name in ["DeepSpeed"]:
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            for handler in handlers:
                logger.addHandler(handler)

    # Set the default logger
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
