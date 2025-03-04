import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fh = logging.FileHandler(filename='./server.log')
formatter = logging.Formatter(
    "[%(asctime)s][%(module)s]%(funcName)s-line:%(lineno)d-%(levelname)s: %(message)s"
)
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)  # Log to console
logger.addHandler(fh)  # Log to file
