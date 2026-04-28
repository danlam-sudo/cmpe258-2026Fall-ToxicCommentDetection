.PHONY: install venv install-venv clean demo-web demo-web-dev demo-web-install demo-web-stop

# Project standard is Python 3.12. Override if needed: `make PYTHON=python3 install-venv`
PYTHON ?= python3.12
DEMO_PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,$(PYTHON))

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

# Install web demo dependencies into active Python
demo-web-install:
	$(DEMO_PYTHON) -m pip install --upgrade pip
	$(DEMO_PYTHON) -m pip install flask transformers torch

# Run Flask DistilBERT web demo (localhost:8000)
demo-web:
	$(DEMO_PYTHON) apps/flask_demo/app.py

# Run Flask DistilBERT web demo in debug mode
demo-web-dev:
	FLASK_DEBUG=1 $(DEMO_PYTHON) apps/flask_demo/app.py

# Stop process listening on demo port (default 8000)
demo-web-stop:
	@PORT_VAL=$${PORT:-8000}; \
	PIDS=$$(lsof -ti tcp:$$PORT_VAL -sTCP:LISTEN || true); \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping process(es) on port $$PORT_VAL: $$PIDS"; \
		kill $$PIDS; \
	else \
		echo "No process listening on port $$PORT_VAL"; \
	fi
