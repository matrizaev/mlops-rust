format:
	cargo fmt --quiet

lint:
	cargo clippy --quiet

test:
	cargo test --quiet

run:
	cargo run $(ARGS)

docker:
	docker build . -t mlops-rust --build-arg model.pkl

docker-run:
	docker run -d -p8080:8080 mlops-rust

test-docker:
	curl -d '{"SepalLengthCm": 1.0, "SepalWidthCm": 1.0, "PetalLengthCm": 1.0, "PetalWidthCm": 1.0}' -H "Content-Type: application/json" -X POST http://localhost:8080/predict

all: format lint test run
