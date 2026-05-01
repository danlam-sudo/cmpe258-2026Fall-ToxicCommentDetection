# Jigsaw Toxic Comment Classification (CMPE258)

Course project workspace for the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

## Prerequisites

- **Python 3.12** — check with `python3.12 --version`. Install via [python.org](https://www.python.org/downloads/), Homebrew (`brew install python@3.12`), or [pyenv](https://github.com/pyenv/pyenv) (this repo includes `.python-version` for pyenv).
- **Git**
- **Make** (optional; macOS/Linux usually have it — Windows users can run the `pip` commands below by hand)

## Repository setup

1. Clone the team repo:

   ```bash
   git clone https://github.com/danlam-sudo/cmpe258-2026Fall-ToxicCommentDetection.git
   cd jigsaw-toxic-comment-classification-challenge
   ```

2. Create and activate a virtual environment:

   **macOS / Linux** (Makefile uses `python3.12` by default)

   ```bash
   make install-venv
   source .venv/bin/activate
   ```

   If `python3.12` is not on your `PATH`, install 3.12 or run `make PYTHON=/path/to/python3.12 install-venv`.

   **Windows (PowerShell)**

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Windows: pip fails on long paths under `.venv\share\jupyter\...`**  
   The full `jupyter` metapackage installs JupyterLab assets with very long filenames. If `pip install` errors with `OSError: [Errno 2] No such file or directory` and a hint about long paths:

   1. **Recommended:** Enable long paths (Windows 10/11): run **PowerShell as Administrator**, then:
      ```powershell
      New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
        -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
      ```
      Reboot (or sign out), then recreate the venv and `pip install` again.
   2. **Or** use a **shorter project path** (e.g. `C:\work\cmpe258`) so `.venv\...` stays under the limit.
   3. **Or** skip the Jupyter metapackage: this repo’s `requirements.txt` is enough for **Cursor / VS Code notebooks** (they use **ipykernel** only). To add the classic Jupyter CLI afterward: enable long paths, then `pip install -r requirements-jupyter.txt`.

3. **Data**: Download the competition files from Kaggle and place these in the `data/` folder:

   - `train.csv`
   - `test.csv`
   - `test_labels.csv`
   - `sample_submission.csv`

   CSVs are **not** committed (they are large). Everyone keeps their own copy locally.

### Optional GloVe embeddings for CNN

The CNN notebook can optionally initialize its embedding layer from **GloVe 6B 100d** when `USE_GLOVE = True`.

1. Download `glove.6B.zip` from the Stanford NLP GloVe page: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
2. Unzip it.
3. Place this file in the project data folder:

   ```text
   data/glove.6B.100d.txt
   ```

The file should keep this exact name unless you also update `GLOVE_PATH` in `notebooks/01_cnn_glove.ipynb`.

`data/glove.6B.100d.txt` is ignored by git because it is large. Each teammate should download and place it locally if they want to run the GloVe CNN experiment.

## Makefile targets

| Target         | Description                                      |
|----------------|--------------------------------------------------|
| `make install` | Upgrade pip and install `requirements.txt` (uses `python3.12` by default — activate your venv first). |
| `make venv`    | Create `.venv` with `python3.12`.                |
| `make install-venv` | Create `.venv` and install dependencies (Unix/macOS). |
| `make demo-web-install` | Install Flask demo deps into active Python. |
| `make demo-web` | Run DistilBERT Flask UI on `http://127.0.0.1:8000`. |
| `make demo-web-dev` | Run Flask UI with debug mode enabled.      |
| `make PYTHON=python3 …` | Override interpreter if needed.           |
| `make clean`   | Delete `.venv`.                                  |

## Working with notebooks

With the venv activated:

```bash
jupyter lab
# or
jupyter notebook
```

Open `eda/EDA.ipynb` for exploratory analysis. **Starter model notebooks** (CNN, BiLSTM, BERT) live in `notebooks/`; DistilBERT notebooks are in `notebooks/distilbert/`. Shared metrics are in `notebooks/metrics_helpers.py`. Shared preprocessing lives under `preprocessing/` (see `preprocessing/README.md`).

## DistilBERT web demo

With `.venv` activated:

```bash
make install
make demo-web
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

By default, the demo loads this bundle:

- `notebooks/distilbert/distilbert_exp07_demo_bundle`

To override the model bundle path:

```bash
BUNDLE_DIR=/absolute/path/to/distilbert_exp07_demo_bundle make demo-web
```

To run on another port:

```bash
PORT=5050 make demo-web
```

### Cursor / VS Code: Run All

The notebook must use the **same Python as `.venv`** (this project targets **3.12**; see `.python-version`). If **Run All** fails with `ModuleNotFoundError: matplotlib` and the traceback shows `site-packages` under **`python3.9`**, the wrong kernel is selected.

1. **Command Palette** → `Python: Select Interpreter` → choose `./.venv/bin/python` (3.12).
2. In the notebook toolbar, open the **kernel picker** (top right) and pick that interpreter / **Python 3.12.x ('.venv')**.
3. If `.venv` was created with an older Python, recreate it: `make clean && make install-venv`, then `source .venv/bin/activate` and `pip install -r requirements.txt`.

Editor settings under `.vscode/` are **not** committed (see `.gitignore`). You can still create `.vscode/settings.json` locally with `"python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"` if you want Cursor/VS Code to default to the project venv.

## Collaboration (lightweight)

- Use **short-lived branches** and **pull requests** into `main` so everyone can review changes.
- Commit **code and notebooks**; keep **data and big artifacts** out of git (see `.gitignore`).

## Adding dependencies

1. `pip install <package>`
2. `pip freeze | grep -i <package>` (or add the line manually with a pinned version)
3. Append to `requirements.txt` and commit so teammates stay in sync.
