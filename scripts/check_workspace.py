"""Validate notebook structure and syntax without executing model training."""
import ast
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            source = TransformerManager().transform_cell(cell.source)
            ast.parse(source, filename=f"{path.name}:cell-{index}")
    print(f"Validated {path.name}")
for path in sorted((ROOT / "scripts").glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Validated {path.name}")
