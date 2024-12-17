"""Bespoke logging functions for use in the other modules.

* color settings for debug (blue), info (green), warning (orange), error (red) and critical (red background) messages (works currently for Idle and Thonny)
* counting of warnings and errors

The module should be used in the following way in a module:

.. code:: python

import Logger
logger = Logger.get_module_logger(__name__, level=0) # In the call of the application module, set the level to the desired logging.xyz,
  e.g. logging.DEBUG, which then sets the log level on all sub-modules

To access error counting results use e.g.

.. code:: python

print("Count:", logger.handlers[0].get_count( ["ERROR"]))

To write a logger warning use

.. code:: python

logger.warning("A message string. Note that only string concatenation with + works")
# logger.info, logger.debug, logger.error and logger.critical work similarly
"""

import logging
import sys


class MsgCounterHandler(logging.StreamHandler):
    levelcount: dict[str, int] = {}

    def __init__(self, logger, *args, **kwargs):
        self.logger = logger
        try:
            self._out = sys.stdout.shell  # type: ignore[union-attr]
        except AttributeError:
            try:
                self._out = sys.stdout
            except Exception:
                self._out = sys.stderr
        super(MsgCounterHandler, self).__init__(*args, **kwargs)
        self.levelcount = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0}
        self.ideType = None
        if "idlelib.run" in sys.modules.keys():  ## if idlelib.run exists
            self.ideType = "Idle"
            self.levelColor = {
                "DEBUG": "DEFINITION",  # blue
                "INFO": "STRING",  # green
                "WARNING": "KEYWORD",  # orange
                "ERROR": "stderr",  # red
                "CRITICAL": "ERROR",
            }  # red background (not yet fully implemented)
        elif (
            "thonny.tktextext" in sys.modules.keys()
        ):  # use escape codes. See also https://en.wikipedia.org/wiki/ANSI_escape_code
            self.ideType = "Thonny"
            self.levelColor = {
                "DEBUG": "\033[34m",  # BLUE
                "INFO": "\033[32m",  # GREEN
                "WARNING": "\033[33m",  # YELLOW
                "ERROR": "\033[31m",  # RED
                "CRITICAL": "\033[37;41m",
            }  # red background, white text
            # all colors: 30:black, 31:red, 32:green, 33:yellow, 34:blue, 35:magenta, 36:cyan, 37: white. Similar: 4x:Background, 9x:bright foreground, 10x:bright background

    def emit(self, record: logging.LogRecord):
        level = record.levelname
        if level not in self.levelcount:
            self.levelcount[level] = 0
        self.levelcount[level] += 1
        # print( record.__dict__) #super().emit( record)
        fullMsg = (
            record.__dict__["filename"].partition(".")[0]
            + " "
            + record.__dict__["levelname"]
            + ": "
            + record.__dict__["msg"]
            + "\n"
        )
        if self.ideType == "Idle":
            self._out.write(fullMsg, self.levelColor[level])
        elif self.ideType == "Thonny":
            self._out.write(self.levelColor[level] + fullMsg + "\033[m")
        else:
            ## TextIOWrapper.write() takes one argument hence self.levelColor[l] removed from the argument
            self._out.write(fullMsg)

    def get_count(self, levels: tuple = ("WARNING", "ERROR"), pretty_print: bool = False):
        if pretty_print:  # return a message string
            msg = "Logger counts"
            for i, level in enumerate(levels):
                if i == 0:
                    msg += ". "
                else:
                    msg += ", "
                if level in self.levelcount:
                    msg += level + "s:" + str(self.levelcount[level])
            return msg
        elif len(levels) > 1:  # return the raw numbers as dictionary
            return self.levelcount
        elif len(levels) == 1:  # return only the requested number
            return self.levelcount[levels[0]]


def get_module_logger(mod_name: str, level: int = logging.DEBUG):
    #    print("Installing logger " +mod_name +" on level " +str(level))
    logger = logging.getLogger(mod_name)
    logger.setLevel(level)
    if len(logger.handlers) == 0:
        handler = MsgCounterHandler(logger)
        handler.setLevel(level)
        # filename = os.path.basename(__file__).partition(".")[0]
        formatter = logging.Formatter(
            "%(name)s. %(levelname)s - %(message)s",
        )
        #             '%(asctime)s %(name)-12s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    #    print("LOGGER " +mod_name +" on level " +str(level) +" installed:")
    return logger


"""
logger = get_module_logger('RP-log')
logger.info("This is the counting logger '" +logger.name +"'")
logger.warning("This is a warning")
logger.error("An error looks like that")
print("Count:", logger.handlers[0].get_count( ["ERROR"]))
"""
