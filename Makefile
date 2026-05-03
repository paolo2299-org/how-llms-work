IMAGE_NAME = how-llms-work
ENV_FILE ?= $(shell if [ -f .env ]; then echo .env; else echo .env.example; fi)
COMPOSE = ENV_FILE=$(ENV_FILE) docker compose -f compose.yml -f compose.dev.yml
COMPOSE_PROD = ENV_FILE=$(ENV_FILE) docker compose -f compose.yml -f compose.prod.yml

# Local development
dev:
	$(COMPOSE) up --build how-llms-work

build:
	docker build -t $(IMAGE_NAME) .

run:
	$(COMPOSE) up --build how-llms-work

test:
	$(COMPOSE) run --rm test

down:
	$(COMPOSE) down --remove-orphans

shell:
	$(COMPOSE) run --rm --service-ports how-llms-work /bin/sh

# Production
prod-start:
	$(COMPOSE_PROD) up -d how-llms-work

prod-stop:
	$(COMPOSE_PROD) stop how-llms-work

prod-restart:
	$(COMPOSE_PROD) restart how-llms-work

.PHONY: dev build run test down shell prod-start prod-stop prod-restart
