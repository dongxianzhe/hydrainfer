import logging
import sys

class NewLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg

level = logging.INFO
root_logger = logging.getLogger()
root_logger.setLevel(level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(level)

formatter = NewLineFormatter(
    fmt="%(levelname)s %(asctime)s[%(filename)s:%(lineno)d] %(message)s", 
    datefmt="%m-%d %H:%M:%S", 
)
handler.setFormatter(formatter)
root_logger.addHandler(handler)

def getLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger
