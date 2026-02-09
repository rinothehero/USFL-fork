import asyncio
import time
import traceback
from functools import wraps

from utils.log_utils import vprint

from ....ws.connection import DisconnectedError


def disconnect_handler(
    retries=3, delay=2, backoff=2, exceptions=(Exception, DisconnectedError)
):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _retries = retries
            _delay = delay
            while _retries > 0:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    vprint(f"Exception: {e}", 0)
                    _retries -= 1
                    if _retries == 0:
                        vprint(
                            f"Function {func.__name__} failed after {retries} retries.", 0
                        )
                        traceback.print_exc()
                        raise e
                    else:
                        vprint(
                            f"Retrying {func.__name__} in {_delay} seconds... ({_retries} retries left)", 2
                        )
                        await asyncio.sleep(_delay)
                        _delay *= backoff

        def sync_wrapper(*args, **kwargs):
            _retries = retries
            _delay = delay
            while _retries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    vprint(f"Exception: {e}", 0)
                    _retries -= 1
                    if _retries == 0:
                        vprint(
                            f"Function {func.__name__} failed after {retries} retries.", 0
                        )
                        traceback.print_exc()
                        raise e
                    else:
                        vprint(
                            f"Retrying {func.__name__} in {_delay} seconds... ({_retries} retries left)", 2
                        )
                        time.sleep(_delay)
                        _delay *= backoff

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
