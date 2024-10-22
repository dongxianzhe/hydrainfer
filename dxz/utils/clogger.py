from datetime import datetime
import sys
import functools
import logging
import colorlog

clogger = logging.getLogger('color_log')
clogger.setLevel(logging.DEBUG)

clogger_handler = logging.StreamHandler()
clogger_handler.setLevel(logging.DEBUG)
clogger_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }))
file_handler = logging.FileHandler('clogger.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(colorlog.ColoredFormatter(
    "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S"))
clogger.addHandler(clogger_handler)
clogger.addHandler(file_handler)

def debug(s: str):
    global clogger
    clogger.debug(s)

def info(s: str):
    global clogger
    clogger.info(s)

def init_clogger():
    return clogger
