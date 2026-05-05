.PHONY: help up down logs build rebuild clean ps

help:
	@echo "Customer Segmentation API - Docker Commands"
	@echo "==========================================="
	@echo "make up       - Start the application"
	@echo "make down     - Stop the application"
	@echo "make build    - Build the Docker image"
	@echo "make rebuild  - Rebuild the Docker image (no cache)"
	@echo "make logs     - View application logs"
	@echo "make ps       - Show running containers"
	@echo "make clean    - Remove containers, volumes, and images"
	@echo "make shell    - Access container shell"

up:
	@echo "Starting Customer Segmentation API..."
	docker-compose up -d
	@echo "✓ Application is running at http://localhost:8000"

down:
	@echo "Stopping Customer Segmentation API..."
	docker-compose down

build:
	@echo "Building Docker image..."
	docker-compose build

rebuild:
	@echo "Rebuilding Docker image (no cache)..."
	docker-compose build --no-cache

logs:
	docker-compose logs -f app

ps:
	docker-compose ps

clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "✓ Cleanup complete"

shell:
	docker-compose exec app /bin/bash

restart:
	@echo "Restarting application..."
	docker-compose restart
	@echo "✓ Application restarted"
