import argparse
from pathlib import Path

from . import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and examine the dataset.")
    parser.add_argument("root", type=Path, help="the dataset root path")
    args = parser.parse_args()
    dataset = Dataset(args.root)
    dataset.print()
