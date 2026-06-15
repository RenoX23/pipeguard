up:
	docker-compose -f infra/docker-compose.yml up -d

down:
	docker-compose -f infra/docker-compose.yml down

restart:
	docker-compose -f infra/docker-compose.yml restart

logs:
	docker-compose -f infra/docker-compose.yml logs -f airflow-scheduler

ps:
	docker-compose -f infra/docker-compose.yml ps

test:
	pytest tests/ -v
