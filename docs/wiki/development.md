# Development

TabFORGE's source checkout contains the package, tests, runnable tutorials, and static documentation. Use Python 3.10 or later and keep changes focused on the affected workflow.

## Set up the environment

Clone the repository and install the development and notebook dependencies:

```bash
git clone https://github.com/SilenceX12138/TabFORGE.git
cd TabFORGE
pip install -e '.[dev,tutorials]'
```

## Repository layout

```text
src/tabforge/
├── api/                 # Public estimators and shared fitted lifecycle
├── config/              # Immutable configuration objects
├── feature_processing/  # Schema capture and reconstruction
├── embeddings/          # Structure-aware Feature Encoder adapter
├── models/              # Score-based Diffusion Transformer and Denoising-aligned Latent Decoder
├── training/            # Optimization, validation, and logging
├── checkpoints/         # Fitted and backbone persistence
└── cli/                 # File-based command-line workflows

docs/wiki/               # Static documentation site and Markdown pages
docs/tutorial/           # Notebook and command-line examples
tests/                   # Estimator and component regression tests
scripts/utils/           # Pylint and Flake8 style check
```

## Tests and style

Run the repository tests and its Pylint/Flake8 style workflow:

```bash
pytest -q tests
bash scripts/utils/style_check.sh
```

## Preview and edit the wiki

The wiki is a static site. Its Markdown pages are loaded by `app.js`; `styles.css` defines the layout and themes. There is no Sphinx build step. Serve the directory locally:

```bash
python -m http.server 8000 \
  --directory docs/wiki
```

Open [localhost:8000](http://localhost:8000/) and stop the server with Ctrl+C. Choose another port by changing the script argument if 8000 is occupied. The header search finds content across all documentation; the sidebar filter narrows the navigation. Press **Ctrl+K** or **/** to open search.

Keep code fences tagged with their language (`python`, `bash`, `json`, `text`, `bibtex`, or `mermaid`). The renderer highlights source code and adds copy buttons. Use responsive process cards or reference tables for architecture overviews and class maps. Detailed Mermaid diagrams retain natural-size labels in a scrollable viewport with an **Expand diagram** button. Model architecture illustrations use the original vector figure from the paper. Add new pages to the curated navigation in `app.js` and the page list in `metadata.json`. Browser renderers, styles, and their licenses live in `vendor/`, so local rendering works without CDN access.

The GitHub Pages workflow uploads `docs/wiki` directly on pushes to `main` or `master`. Styles, scripts, and figures use relative paths so the site works under a repository URL prefix.

## Notebook development

Start Jupyter from the checkout after installing the notebook dependencies:

```bash
jupyter notebook docs/tutorial/notebooks
```

The [tutorial gallery](tutorials.md) lists all examples and describes Colab setup. For lightweight verification, use mock embeddings with `checkpoint=None`; use the pretrained Structure-aware Feature Encoder for research results.

## Build the package

The project uses Hatchling through the standard Python build interface:

```bash
pyproject-build
```

Artifacts are written to `dist/` under the distribution name `tabforge-ml`. The wheel contains the `tabforge` package; the source distribution also includes documentation, tests, and the style-check script. The package version is declared in `pyproject.toml` and `src/tabforge/_metadata.py`; update both together when preparing a release.

## Extend a workflow

Trace task-specific behavior in [the estimators](public_estimator_api_estimators.md) and shared fitting, sampling, and persistence in [the lifecycle core](public_estimator_api_lifecycle.md). Schema changes must stay aligned with [feature processing](tabular_feature_processing.md) and [encoder token layout](tabpfn_embedding_adapter.md). Training policy belongs in [training orchestration](training_orchestration.md). Add focused coverage to the corresponding existing tests and run the style workflow before submitting changes.
