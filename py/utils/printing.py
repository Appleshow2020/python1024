import sys
import time
from enum import Enum
from typing import TextIO


class LT(Enum):
    error = ('error', '\033[31m')
    success = ('success', '\033[32m')
    warning = ('warning', '\033[33m')
    info = ('info', '\033[34m')
    debug = ('debug', '\033[90m')

    def __init__(self, label: str, color: str):
        self.label = label
        self.color = color


class _Printf:
    @staticmethod
    def _format(
        values: tuple,
        ptype: LT,
        sep: str,
        simulation: bool,
        useReset: bool
    ) -> str:

        timestamp = time.strftime('%X')

        prefix = (
            f'{ptype.color}[{timestamp}]'
            f'{" [SIMULATION]" if simulation else ""}'
            f' [{ptype.name.upper()}] '
        )

        message = sep.join(map(str, values))

        if useReset:
            message += '\033[0m'

        return prefix + message

    @staticmethod
    def _write(
        message: str,
        end: str,
        file: TextIO,
        flush: bool
    ) -> None:

        file.write(message + end)

        if flush:
            file.flush()


def printf(
    *values: object,
    ptype: LT = LT.debug,
    sep: str | None = ' ',
    end: str | None = '\n',
    file: TextIO | None = None,
    flush: bool = False,
    useReset: bool = True,
    simulation: bool = False
) -> None:

    if not isinstance(ptype, LT):
        raise TypeError('ptype must be an instance of LT')

    if sep is not None and not isinstance(sep, str):
        raise TypeError('sep must be None or a string')

    if end is not None and not isinstance(end, str):
        raise TypeError('end must be None or a string')

    if file is None:
        file = sys.stdout

    if sep is None:
        sep = ' '

    if end is None:
        end = ''

    message = _Printf._format(
        values,
        ptype,
        sep,
        simulation,
        useReset
    )

    _Printf._write(message, end, file, flush)