format:
	cargo fmt --quiet

lint:
	cargo clippy --quiet

test:
	cargo test --quiet

run:
	cargo run $(ARGS)

docker:
	docker build . -t ghcr.io/matrizaev/mlops-rust:main

docker-run:
	docker run -d -p8080:8080 ghcr.io/matrizaev/mlops-rust:main

docker-test:
	curl -d '{"SepalLengthCm": 1.0, "SepalWidthCm": 1.0, "PetalLengthCm": 1.0, "PetalWidthCm": 1.0}' -H "Content-Type: application/json" -X POST http://localhost:8080/predict

all: format lint test run
