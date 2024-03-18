format:
	cargo fmt --quiet

lint:
	cargo clippy --quiet

test:
	cargo test --quiet

run:
	cargo run 

docker:
	docker build . -t mlops-rust

test-docker:
	curl -d '{"SepalLengthCm": 1.0, "SepalWidthCm": 1.0, "PetalLengthCm": 1.0, "PetalWidthCm": 1.0}' -H "Content-Type: application/json" -X POST http://localhost:8080/predict

all: format lint test run
