# Makefile shortcuts for HRM
# Usage: make <target>

PY?=python3
VENV?=.venv
ACTIVATE=source $(VENV)/bin/activate

.PHONY: env tiny sudoku train-tiny train-sudoku eval clean

env:
	$(PY) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt && pip install numpy

# Build tiny synthetic dataset
tiny:
	$(ACTIVATE) && ./scripts/build_tiny_dataset.sh

# Build 1k augmented sudoku dataset
sudoku:
	$(ACTIVATE) && ./scripts/build_sudoku_1k.sh

# Train tiny demo (quick smoke test)
train-tiny:
	$(ACTIVATE) && ./scripts/train_tiny_demo.sh

# Train small sudoku model (override vars inline, e.g. make train-sudoku EPOCHS=500 )
train-sudoku:
	$(ACTIVATE) && ./scripts/train_sudoku_small.sh

# Evaluate a checkpoint: make eval CHK=checkpoints/.../step_x
CHK?=
ifeq ($(CHK),)
	eval:
		@echo "Set CHK=path/to/checkpoint" && exit 1
else
	eval:
		$(ACTIVATE) && ./scripts/eval_checkpoint.sh $(CHK)
endif

clean:
	rm -rf $(VENV) build dist __pycache__ */__pycache__
