import contextlib
import logging


class IndentFilter(logging.Filter):
    indent = 0  # class-level, shared across all loggers using this filter
    first_log = True  # class-level, shared across all loggers using this filter

    def filter(self, record: logging.LogRecord) -> bool:
        if IndentFilter.first_log and IndentFilter.indent > 0:
            start = "|_ "
            IndentFilter.first_log = False
        else:
            start = "│  "
        if IndentFilter.indent > 0:
            prefix = "   " * (IndentFilter.indent - 1) + start
        else:
            prefix = ""
        record.msg = f"{prefix}{record.msg}"
        return True


@contextlib.contextmanager
def indented():
    IndentFilter.indent += 3
    IndentFilter.first_log = True
    try:
        yield
    finally:
        IndentFilter.indent -= 3


def make_bar_format(desc: str) -> str:
    prefix = (
        "   " * (IndentFilter.indent - 1) + "|_ " if IndentFilter.indent > 0 else ""
    )
    return f"{prefix}{desc}: {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}]"
