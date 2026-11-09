"""Export a retained QUEST checkpoint and tokenizer; never retrain or download weights."""

import argparse
import json
from pathlib import Path

import pandas as pd

from quest.serving import export_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--example-csv", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
print(
    json.dumps(
        export_model(args.run_dir, args.output, pd.read_csv(args.example_csv).head(2)), indent=2
    )
)
