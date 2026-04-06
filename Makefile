.PHONY: setup run api test lint format clean docker-up docker-down help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and pre-commit hooks
	pip install -r requirements.txt
	pre-commit install

run: ## Run the Streamlit app
	streamlit run src/app.py --server.port=8501

api: ## Run the FastAPI server
	uvicorn src.api:app --reload --port 8000

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term -k "not integration"

test-all: ## Run all tests including integration
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean: ## Remove generated files
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-up: ## Start all services with Docker
	docker-compose up --build -d

docker-down: ## Stop all Docker services
	docker-compose down

pull-model: ## Pull default Ollama model (Mistral 7B)
	ollama pull mistral
