.PHONY: help build clean install test style docs-serve docs-build docs-clean init release clear-branch launch-server translate-args sweep-make sweep-prod tags rebase-public interactive

# Default target
help:
	@echo "Available targets:"
	@echo "  build         		Build the package"
	@echo "  clean         		Clean build artifacts"
	@echo "  install      		Install the package locally in editable mode"
	@echo "  style		        Run style checks (flake8, pylint)"
	@echo "  docs-serve        	Build and deploy documentation"
	@echo "  docs-build			Build the documentation"
	@echo "  docs-clean  		Clean documentation build"
	@echo "  init    			Init the project environment"
	@echo "  release      		Release the package with squashed commits"
	@echo "  clear-branch 		Clean deprecated git branches"
	@echo "  launch-server      Launch the local workflow server"
	@echo "  translate-args     Translate command-line arguments to VSCode json"
	@echo "  sweep-make			Make a sweep configuration file"
	@echo "  sweep-prod			Launch the sweep in the production environment"
	@echo "  tags				Generate tags for the project"
	@echo "  rebase-public		Rebase the current branch onto the latest public master branch"
	@echo "  interactive		Launch the interactive development environment"

# Build and clean when src is not empty
build: clean
	if [ -z "$$(ls -A src 2>/dev/null)" ]; then \
		echo "No source files found in 'src' directory. Skipping build."; \
	else \
		echo "Building the package..."; \
		python -m build; \
	fi

clean:
	bash scripts/dev/clean_build.sh

# Installation
install:
	if [ -z "$$(ls -A src 2>/dev/null)" ]; then \
		uv pip install -r pyproject.toml; \
	else \
		make build; \
		uv pip install -e .; \
	fi

# Code quality
style:
	bash scripts/utils/style_check.sh

# Documentation
docs-serve: docs-clean
	bash scripts/doc/server.sh

docs-build: docs-clean
	cd docs/wiki && make html

docs-clean:
	cd docs/wiki && make clean

# Development
init:
	bash scripts/git/init.sh

release:
	bash scripts/git/release.sh

clear-branch:
	bash scripts/dev/clear_branch.sh

launch-server:
	bash scripts/dev/launch_workflow_server.sh

translate-args:
	bash scripts/dev/translate_args.sh

sweep-make:
	bash scripts/dev/make_sweep.sh

sweep-prod:
	@if command -v screen >/dev/null 2>&1; then \
		bash scripts/dev/execute_sweep_prod.sh; \
	else \
		bash scripts/dev/slurm/submit_sweep_sbatch.sh prod; \
	fi

sweep-dev:
	@if command -v screen >/dev/null 2>&1; then \
		bash scripts/dev/execute_sweep_dev.sh; \
	else \
		bash scripts/dev/slurm/submit_sweep_sbatch.sh dev; \
	fi

tags:
	python scripts/dev/generate_uuid.py

rebase-public:
	bash scripts/git/rebase_public.sh

interactive:
	bash scripts/dev/slurm/gpu_interactive.sh