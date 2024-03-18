FROM rust:latest as build-env
WORKDIR /app
COPY . /app
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12
COPY --from=build-env /app/target/release/mlops-rust /
EXPOSE 8080
ARG model_path
CMD ["./mlops-rust", "--model-path", ${model_path}, "serve"]
