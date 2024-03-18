use crate::model::{json_to_ndarray, load_model, CustomTrainedModel};
use crate::web::web::Data;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use linfa::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct PredictionPayload {
    SepalLengthCm: f64,
    SepalWidthCm: f64,
    PetalLengthCm: f64,
    PetalWidthCm: f64,
}

//create a function that returns a hello world
#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Iris prediction service")
}

//create a function that returns a 200 status code
#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok()
}

//create a post function that runs the model and returns the prediction
#[post("/predict")]
async fn predict(
    data: Data<CustomTrainedModel>,
    payload: web::Json<PredictionPayload>,
) -> impl Responder {
    let model = data.as_ref();
    let json_value = serde_json::to_string(&payload).unwrap();
    let features_ndarray = json_to_ndarray(&json_value).unwrap();
    let pred = model.predict(&features_ndarray);
    let str = serde_json::to_string(&pred).unwrap();
    HttpResponse::Ok().body(str)
}

//create a function that returns the version of the service
#[get("/version")]
async fn version() -> impl Responder {
    //print the version of the service
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    HttpResponse::Ok().body(env!("CARGO_PKG_VERSION"))
}

#[actix_web::main]
pub async fn serve(model_path: &str, bind_address: &Option<String>) -> std::io::Result<()> {
    //add a print message to the console that the service is running
    println!("Running the service");
    let model = load_model(model_path);
    let data = Data::new(model);
    let service = HttpServer::new(move || {
        App::new()
            .app_data(Data::clone(&data))
            .service(hello)
            .service(health)
            .service(version)
            .service(predict)
    });
    match bind_address {
        Some(address) => {
            println!("Binding to address: {}", address);
            service.bind(address)?.run().await
        }
        None => {
            println!("No address provided, binding to default address");
            service.bind("0.0.0.0:8080")?.run().await
        }
    }
}
