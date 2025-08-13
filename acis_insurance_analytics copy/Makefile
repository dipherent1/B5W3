.PHONY: setup eda test models clean

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

eda:
\tpython -m src.eda.eda --config configs/config.yaml

test:
\tpython -m src.tests.hypothesis_tests --config configs/config.yaml

models:
\tpython -m src.models.train_all --config configs/config.yaml

clean:
\trm -rf data/intermediate/* data/processed/* reports/* models/*
\ttouch reports/.gitkeep reports/figures/.gitkeep
