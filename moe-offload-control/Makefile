.PHONY: install test lint clean

install:
	pip install -e .

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
