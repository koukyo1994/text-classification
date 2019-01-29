import sys
import time
import logging
import datetime as dt

from pathlib import Path
from contextlib import contextmanager


@contextmanager
def timer(name, logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0:.0f} s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_logger(name="Main", exp="exp", log_dir="log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    log_path = path / (
        exp + "_" + dt.datetime.now().strftime('%Y%m%d%H%M%S') + ".log")
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
