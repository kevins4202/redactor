"""Download the official DAVIS 2017 train/validation set (480p)."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile


URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--sequences", nargs="+", help="Extract only these sequences; the full archive is cached")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / "downloads" / "DAVIS-2017-trainval-480p.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        total = 0
        with urlopen(URL, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                total += len(chunk)
                print(f"Downloaded {total / 1024**2:.0f} MiB", flush=True)
        partial.replace(archive)
    root = args.destination.resolve()
    with zipfile.ZipFile(archive) as source:
        members = []
        for member in source.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive contains a path outside the destination")
            parts = Path(member.filename).parts
            if args.sequences and len(parts) >= 4 and parts[1] in ("JPEGImages", "Annotations"):
                if parts[3] not in args.sequences:
                    continue
            members.append(member)
        source.extractall(root, members=members)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {"url": URL, "sha256": digest, "sequences": args.sequences or "all", "resolution": "480p"}
    (root / "DAVIS" / "download.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"DAVIS is ready at {root / 'DAVIS'}")


if __name__ == "__main__":
    main()
