# ==== Config ====
IMAGE_NAME = stroke-api
TAG = latest
CONTAINER_NAME = stroke-api
MODEL_DIR = ./models
PORT = 8000

# ==== Commands ====

# Build docker image
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Build & run using docker compose
up:
	docker compose up -d

# Stop containers
down:
	docker compose down

# View logs (follow)
logs:
	docker compose logs -f

# Run interactively without compose (for debugging)
run:
	docker run --rm -it \
		-p $(PORT):8000 \
		-v $(MODEL_DIR):/models:ro \
		-e MODEL_PATH=/models/my_model.joblib \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(TAG)

# Stop container by name
stop:
	docker stop $(CONTAINER_NAME) || true

# Remove stopped containers & images
clean:
	docker compose down --rmi local --volumes --remove-orphans

# Check health
health:
	curl -s http://localhost:$(PORT)/health | jq

# Predict sample
predict:
	curl -s http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"points":[{"x":0,"y":0,"pressure":0.01},{"x":5,"y":0,"pressure":0.2},{"x":10,"y":0,"pressure":0.02}]}' | jq
