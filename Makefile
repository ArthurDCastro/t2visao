PYTHON := python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: all download setup clean

all: setup
	$(PY) main.py

download: setup
	$(PY) download-images.py

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV)