"""List and optionally safely extract the observed course Project4 ZIP."""
import argparse
from collections import Counter
from pathlib import Path
import stat
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--extract-to")
    args = parser.parse_args()
    archive = Path(args.archive)
    if archive.suffix != ".zip":
        raise SystemExit("A complete .zip archive is required; partial downloads cannot be extracted")
    with zipfile.ZipFile(archive) as source:
        files = source.infolist()
        print(f"{len(files)} entries; {sum(f.file_size for f in files)/1024**3:.3f} GiB uncompressed")
        top = {}
        for file in files:
            first = file.filename.replace("\\", "/").split("/")[0]
            top[first] = top.get(first, 0) + 1
        print("Top-level entries:", top)
        metadata = Counter()
        for file in files:
            if file.filename.lower().endswith((".json", ".txt", ".bin", ".csv")):
                parent = str(Path(file.filename).parent)
                metadata[parent] += 1
                if metadata[parent] <= 2:
                    print(file.filename, file.file_size)
        print("Metadata entries per directory:", dict(metadata))
        if args.extract_to:
            destination = Path(args.extract_to).resolve()
            for file in files:
                target = (destination / file.filename.replace("\\", "/")).resolve()
                if not target.is_relative_to(destination) or stat.S_ISLNK(file.external_attr >> 16):
                    raise ValueError(f"Unsafe ZIP member: {file.filename}")
                if target.exists() and not file.is_dir():
                    raise FileExistsError(f"Will not overwrite existing file: {target}")
            bad = source.testzip()
            if bad:
                raise ValueError(f"ZIP CRC error: {bad}")
            source.extractall(destination)
            print("Extracted:", destination)


if __name__ == "__main__":
    main()
