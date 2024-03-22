use crate::model::{json_to_ndarray, load_model, CustomTrainedModel};
use actix_web::web::Data;
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder};
use actix_web::middleware::Logger;
use linfa::prelude::*;

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
async fn predict(data: Data<CustomTrainedModel>, text: String) -> impl Responder {
    let model = data.as_ref();
    match json_to_ndarray(&text) {
        Ok(features) => {
            let pred = model.predict(&features);
            let str = serde_json::to_string(&pred).unwrap();
            HttpResponse::Ok().body(str)
        }
        Err(e) => HttpResponse::BadRequest().body(e.to_string()),
    }
}

//create a function that returns the version of the service
#[get("/version")]
async fn version() -> impl Responder {
    //print the version of the service
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    HttpResponse::Ok().body(env!("CARGO_PKG_VERSION"))
}

#[actix_web::main]
pub async fn serve(model_path: &str, bind_address: Option<&str>) -> std::io::Result<()> {
    //add a print message to the console that the service is running
    println!("Running the service");
    let model = load_model(model_path);
    let data = Data::new(model);
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    let service = HttpServer::new(move || {
        App::new()
            .app_data(Data::clone(&data))
            .wrap(Logger::default())
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
