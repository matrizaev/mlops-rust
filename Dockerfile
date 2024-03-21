FROM rust:latest as build-env
WORKDIR /app
COPY . /app
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12
COPY --from=build-env /app/target/release/mlops-rust /
COPY settings.yaml /model.pkl
EXPOSE 8080

CMD ["./mlops-rust", "serve"]
