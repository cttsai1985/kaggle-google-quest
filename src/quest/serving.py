"""Package retained text processing and weights for question/answer scoring."""

import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import torch
from mlflow.models import infer_signature
from transformers import AutoTokenizer

from quest.data import TARGETS, TEXT_COLUMNS, TokenizedPairs, sha256
from quest.runtime import load_model, probabilities


class QuestionQualityModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        root = Path(context.artifacts["predictor"])
        contract = json.loads((root / "serving-contract.json").read_text())
        files = contract["files"]
        required = {"model.pt", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name != "serving-contract.json"
        }
        if not required <= set(files) or set(files) != actual:
            raise ValueError("Question-quality package is missing verified model/tokenizer files")
        for name, expected in files.items():
            source = root / name
            path = source.resolve()
            if not path.is_relative_to(root) or source.is_symlink() or sha256(path) != expected:
                raise ValueError("Question-quality package checksum mismatch")
        torch.set_num_threads(2)
        self.model, checkpoint = load_model(root, torch.device("cpu"))
        self.config = checkpoint["config"]
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame) or model_input.empty:
            raise ValueError("Provide question_title, question_body and answer text")
        if list(model_input.columns) != list(TEXT_COLUMNS):
            raise ValueError("Text fields/order differ from the model contract")
        for name in TEXT_COLUMNS:
            if not model_input[name].map(lambda value: isinstance(value, str)).all():
                raise ValueError("Text values must be strings, including empty strings")
        dataset = TokenizedPairs(model_input, self.tokenizer, self.config, labeled=False)
        values = probabilities(self.model, dataset, 2, torch.device("cpu"))
        return pd.DataFrame(values, columns=TARGETS)


def export_model(run_dir, output, example):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Serving export already exists; select a fresh output directory")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Export requires a completed transformer run")
    names = [
        "model.pt",
        *[
            str(path.relative_to(run_dir))
            for path in sorted((run_dir / "tokenizer").rglob("*"))
            if path.is_file()
        ],
    ]
    for name in names:
        if manifest["artifacts"].get(name) != sha256(run_dir / name):
            raise ValueError("Producer model or tokenizer checksum mismatch")
    example = example[list(TEXT_COLUMNS)].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quest-export-") as directory:
        bundle = Path(directory) / "predictor"
        bundle.mkdir()
        for name in names:
            destination = bundle / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / name, destination)
        contract = {
            "files": {name: sha256(bundle / name) for name in names},
            "producer_manifest_sha256": sha256(run_dir / "manifest.json"),
            "producer_source": manifest["source"],
            "targets": list(TARGETS),
            "input_fields": list(TEXT_COLUMNS),
        }
        (bundle / "serving-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        code = Path(directory) / "code" / "quest"
        shutil.copytree(
            Path(__file__).parent, code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        predictor = QuestionQualityModel()
        context = type("Context", (), {"artifacts": {"predictor": str(bundle)}})()
        predictor.load_context(context)
        expected = predictor.predict(context, example)
        del predictor
        mlflow.pyfunc.save_model(
            path=str(output),
            python_model=QuestionQualityModel(),
            artifacts={"predictor": str(bundle)},
            code_paths=[str(code)],
            input_example=example,
            signature=infer_signature(example, expected),
            pip_requirements=[
                "--extra-index-url https://download.pytorch.org/whl/cpu",
                *[
                    f"{name}=={importlib.metadata.version(name)}"
                    for name in [
                        "mlflow",
                        "numpy",
                        "pandas",
                        "scipy",
                        "scikit-learn",
                        "torch",
                        "transformers",
                        "safetensors",
                        "sentencepiece",
                    ]
                ],
            ],
            metadata={
                "project_id": "quest",
                "producer_manifest_sha256": contract["producer_manifest_sha256"],
                "input_boundary": "question title, body and answer text",
                "target_count": 30,
            },
        )
    return {
        "status": "exported",
        "targets": len(TARGETS),
        "example_rows": len(example),
        "producer_manifest_sha256": contract["producer_manifest_sha256"],
    }
