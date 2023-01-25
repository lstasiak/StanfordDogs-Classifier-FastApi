export $(shell sed 's/=.*//' .env)

.PHONY: help up start stop restart status ps clean

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Up all or c=<name> containers in foreground
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) up $(c)

up-d: ## Up all or c=<name> containers in background
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) up -d $(c)

start: ## Start all or c=<name> containers
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) start $(c)

build: ## Build all or c=<name> containers in background
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) up --build $(c)

build-d: ## Build all or c=<name> containers in foreground
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) up --build -d $(c)

stop: ## Stop all or c=<name> containers
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) stop $(c)

restart: ## Restart all or c=<name> containers
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) restart $(c)

rebuild: ## Rebuild all or c=<name> containers
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) bash -c "down && up --build -d"

logs: ## Show logs for all or c=<name> containers
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) logs --tail=$(or $(n), 100) -f $(c)

status: ## Show status of containers
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) ps

ps: status ## Alias of status

clean: ## Clean all data
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) down

down: clean ## Alias of clean

prune: ## Prune all unused containers
	docker system prune --all --volumes

images: ## Show all images
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) images

exec: ## Exec container
	docker-compose -f $(or $(DOCKER_COMPOSE_FILE), docker-compose.yml) exec $(or $(c), web) bash

check:
	docker-compose exec $(or $(c), web) pre-commit run $(or $(e), --all-files)

mypy:
	docker-compose  exec $(or $(c), web) mypy $(or $(e), .)

perform: ## Perform code by black, isort
	docker-compose  exec $(or $(c), web) black $(or $(e), .)
	docker-compose  exec $(or $(c), web) isort --profile black $(or $(e), .)

predict: ## Perform prediction using python script
	docker-compose  exec $(or $(c), web) python src/inference.py

train-model: ## Perform model training without saving
	docker-compose  exec $(or $(c), web) python src/model_training.py --epochs $(or $(e), 1) -lr $(or $(e), 10e-4)

train-model-save: ## Perform model training with saving
	docker-compose  exec $(or $(c), web) python src/model_training.py --epochs $(or $(e), 1) -lr $(or $(e), 10e-4) --save

tests:
	docker-compose exec web pytest -k "tests"
