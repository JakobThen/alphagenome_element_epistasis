# Environment

| File | Purpose |
|---|---|
| `requirements.txt` | Direct dependencies, unpinned. Edit this one. |
| `requirements.lock.txt` | Every package pinned to an exact version. One file for Linux, macOS and Windows: platform-specific packages carry markers, so pip installs the right set. |
| `environment.yml` | Conda wrapper providing Python 3.11 and pip. |
| `overrides.txt` | Read only when the lock is regenerated (see below). |

## Install

With pip, Python 3.11 or newer, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --no-deps -r env/requirements.lock.txt
pip install --no-deps -e .
```

With conda:

```bash
conda env create -f env/environment.yml
conda activate element_epistasis
pip install --no-deps -r env/requirements.lock.txt
pip install --no-deps -e .
```

Install with `--no-deps` as
pip re-reads the dependency lists of individual packages and tries to add back
the ones the lock leaves out on some platforms (below).


