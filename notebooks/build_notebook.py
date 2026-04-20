from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_end_to_end_pipeline.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        md(
            "# Smart Agri Assistant (Maharashtra)\n"
            "Production-backed notebook for preprocessing, EDA, model training, and prediction."
        )
    )

    cells.append(md("## Setup"))
    cells.append(
        code(
            "import sys\n"
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n"
            "\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "sys.path.append(str(ROOT))\n"
            "\n"
            "from src.data_pipeline import preprocess\n"
            "from src.model_training import train_and_select\n"
            "from src.predictor import Predictor\n"
        )
    )

    cells.append(md("## Phase 1: Build unified processed dataset"))
    cells.append(
        code(
            "df = preprocess()\n"
            "print(df.shape)\n"
            "df.head()"
        )
    )

    cells.append(md("## Phase 2: Validate dataset coverage"))
    cells.append(
        code(
            "print('Crop classes:', df['crop_label'].nunique())\n"
            "df['crop_label'].value_counts().head(15)"
        )
    )

    cells.append(md("## Phase 3: Train all models and persist artifacts"))
    cells.append(
        code(
            "metrics = train_and_select()\n"
            "print(json.dumps({k: metrics[k] for k in ['model', 'accuracy', 'f1_macro']}, indent=2))"
        )
    )

    cells.append(md("## Phase 4: Review generated plots"))
    cells.append(
        code(
            "visual_dir = ROOT / 'visualizations'\n"
            "for name in ['crop_distribution.png', 'correlation_heatmap.png', 'regional_patterns.png', 'feature_importance.png', 'confusion_matrix.png']:\n"
            "    path = visual_dir / name\n"
            "    if path.exists():\n"
            "        print(name)\n"
            "        display(Image(filename=str(path)))"
        )
    )

    cells.append(md("## Phase 5: Sample prediction"))
    cells.append(
        code(
            "predictor = Predictor()\n"
            "sample = {\n"
            "    'N': 90,\n"
            "    'P': 42,\n"
            "    'K': 55,\n"
            "    'pH': 6.5,\n"
            "    'temperature': 26.0,\n"
            "    'humidity': 82.0,\n"
            "    'rainfall': 210.0,\n"
            "    'soil_type': 'Black',\n"
            "    'season': 'Autumn',\n"
            "    'region': 'Konkan'\n"
            "}\n"
            "predictor.predict_topk(sample)"
        )
    )

    nb["cells"] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f'Notebook written to {NOTEBOOK_PATH}')


if __name__ == "__main__":
    build()
