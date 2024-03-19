use std::path::PathBuf;

use clap::{Parser, Subcommand};
use config::Config;
use serde_derive::Deserialize;

use mlops_rust::model::{download_dataset, read_dataset, train_track_model};
use mlops_rust::web::serve;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long, value_name = "FILE")]
        dataset_path: Option<PathBuf>,
    },
    Serve {},
}

#[derive(Debug, Deserialize)]
struct Settings {
    model_path: String,
    bind_address: String,
    mlflow_tracking_uri: String,
    mlflow_experiment_name: String,
    mlflow_run_name: String,
}

pub fn main() {
    let cfg = Config::builder()
        // Add in `./Settings.toml`
        .add_source(config::File::with_name("settings.yaml"))
        // Add in settings from the environment (with a prefix of APP)
        // Eg.. `APP_DEBUG=1 ./target/app` would set the `debug` key
        .add_source(config::Environment::with_prefix("APP"))
        .build()
        .unwrap();

    let settings: Settings = cfg.try_deserialize().unwrap();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Train { dataset_path } => {
            let path: PathBuf = match dataset_path {
                Some(provided_path) => provided_path.clone(),
                None => download_dataset("scikit-learn/iris", "Iris.csv").unwrap(),
            };

            let dataset = read_dataset(path.to_str().unwrap());

            train_track_model(
                &dataset,
                &settings.mlflow_tracking_uri,
                &settings.mlflow_experiment_name,
                Some(&settings.mlflow_run_name),
            )
            .expect("Unsuccessful training attempt.");
        }
        Commands::Serve {} => match serve(&settings.model_path, Some(&settings.bind_address)) {
            Ok(_) => println!("Server started"),
            Err(e) => println!("Error starting server: {}", e),
        },
    }
}
