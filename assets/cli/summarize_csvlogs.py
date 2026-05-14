import stable_pretraining as spt
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()

    reader = spt.loggers.CSVLogAutoSummarizer(args.path)
    df = reader.collect()
    print(df)
