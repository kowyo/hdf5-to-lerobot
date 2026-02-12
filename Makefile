.PHONY: setup install-hooks check format test clean

# Setup development environment
setup:
	uv sync
	$(MAKE) install-hooks

# Install pre-commit hooks
install-hooks:
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed"

# Run all pre-commit checks on all files
check:
	uv run pre-commit run --all-files

# Run ruff linter and fix issues
lint:
	uv run ruff check --fix .

# Run ruff formatter
format:
	uv run ruff format .

# Clean build artifacts
clean:
	rm -rf dist/ build/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run the pipeline
run:
	uv run python -m hdf5_to_lerobot

download-dataset:
	uv run python scripts/download_dataset.py $(REPO)

# Show help
help:
	@echo "Available targets:"
	@echo "  setup        - Install dependencies and pre-commit hooks"
	@echo "  install-hooks- Install pre-commit hooks only"
	@echo "  check        - Run all pre-commit checks on all files"
	@echo "  lint         - Run ruff linter with auto-fix"
	@echo "  format       - Run ruff formatter"
	@echo "  clean        - Clean build artifacts and cache files"
	@echo "  run          - Run the hdf5_to_lerobot pipeline"
	@echo "  download-dataset REPO=<repo_id> - Download dataset from Hugging Face"
	@echo "  help         - Show this help message"
