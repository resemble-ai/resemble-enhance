import logging
import selectors
import sys
from functools import cache

from .distributed import global_leader_only

_logger = logging.getLogger(__name__)


@cache
def _get_stdin_selector():
    selector = selectors.DefaultSelector()
    selector.register(fileobj=sys.stdin, events=selectors.EVENT_READ)
    return selector


@global_leader_only(boardcast_return=True)
def non_blocking_input():
    s = ""
    selector = _get_stdin_selector()
    events = selector.select(timeout=0)
    for key, _ in events:
        s: str = key.fileobj.readline().strip()
        _logger.info(f'Get stdin "{s}".')
    return s
