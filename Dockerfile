FROM rust:latest as build-env
WORKDIR /app
COPY . /app
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12
ARG model_path
COPY --from=build-env /app/target/release/mlops-rust /
COPY ${model_path} /model.pkl
EXPOSE 8080

CMD ["./mlops-rust", "--model-path", "model.pkl", "serve"]
