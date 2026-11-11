"""Reference-only fold recovery; model inference uses the modern project runtime."""

import csv
import hashlib
import json
import sys
from pathlib import Path


def main():
    if sys.version_info[:2] != (3, 6):
        raise RuntimeError("Use the documented Python-3.6 reference container for index recovery")
    import joblib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.model_selection import GroupKFold

    actual = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    expected = {
        "numpy": "1.18.1",
        "scipy": "1.4.1",
        "scikit-learn": "0.22.1",
        "joblib": "0.13.2",
    }
    if actual != expected:
        raise RuntimeError("Reference numerical versions differ from the author record")
    with Path("/data/train.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    groups = np.array([r["question_title"] for r in rows], dtype=object)
    _, valid = next(GroupKFold(5).split(np.zeros(len(rows)), groups=groups))
    held = set(valid.tolist())
    output = Path("/out/reference-fold0.csv")
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["qa_id", "reconstructed_holdout"])
        for i, row in enumerate(rows):
            writer.writerow([row["qa_id"], i in held])
    proof = {
        "packages": actual,
        "rows": len(rows),
        "fold0_rows": len(held),
        "groups": "question_title",
        "source_train_sha256": hashlib.sha256(Path("/data/train.csv").read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "scope": "Author-version reconstruction; original saved IDs are unavailable.",
    }
    Path("/out/reference-folds.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
