.PHONY: install venv install-venv clean

# Project standard is Python 3.12. Override if needed: `make PYTHON=python3 install-venv`
PYTHON ?= python3.12

# Install dependencies into the active Python (use after activating a venv)
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Create .venv (Unix/macOS)
venv:
	$(PYTHON) -m venv .venv
	@echo "Activate: source .venv/bin/activate"

# Create venv and install into it (Unix/macOS)
install-venv:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Activate: source .venv/bin/activate"

# Remove local venv
clean:
	rm -rf .venv
