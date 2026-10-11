"""Command-line entry point; baseline helpers remain importable for comparisons."""

from redactor.baseline import segment, translate
from redactor.cli import main, parse_args
from redactor.render import redact_frame as blur
from redactor.ui import choose_mask, draw_shape


def track(previous_frame, frame, mask):
    return translate(previous_frame, frame, mask)[0]


if __name__ == "__main__":
    raise SystemExit(main())
