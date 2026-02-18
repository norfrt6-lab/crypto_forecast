.PHONY: install test lint format clean

install:
	pip install -r requirements.txt

test:
	pytest --tb=short -q

lint:
	flake8 src/
	mypy src/

format:
	isort src/
	black src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .mypy_cache
