"""Fetch the official SAM 2.1 tiny checkpoint with a recorded checksum."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"


def main():
    path = Path("models/sam2.1_hiera_tiny.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".partial")
        with urlopen(URL, timeout=120) as response, partial.open("wb") as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
        partial.replace(path)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    path.with_suffix(".json").write_text(json.dumps({"url": URL, "sha256": checksum}, indent=2) + "\n")
    print(f"Ready: {path}\nSHA256: {checksum}")


if __name__ == "__main__":
    main()
