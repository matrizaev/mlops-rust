use std::path::PathBuf;

use clap::{Parser, Subcommand};
use linfa::prelude::*;

use mlops_rust::model::{
    download_dataset, json_to_ndarray, load_model, read_dataset, save_model, train_model,
};

use mlops_rust::web::serve;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, value_name = "FILE")]
    model_path: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long, value_name = "FILE")]
        dataset_path: Option<PathBuf>,
    },
    Predict {
        #[arg(short, long, value_name = "JSON")]
        json_value: String,
    },
    Serve {
        #[arg(short, long, value_name = "0.0.0.0:8080")]
        bind_address: Option<String>,
    },
}

pub fn main() {
    let cli = Cli::parse();

    println!("Model path: {:?}", cli.model_path);

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd

    match &cli.command {
        Commands::Train { dataset_path } => {
            let path: PathBuf = match dataset_path {
                Some(provided_path) => provided_path.clone(),
                None => download_dataset("scikit-learn/iris", "Iris.csv").unwrap(),
            };

            let dataset = read_dataset(path.to_str().unwrap());
            let (train, valid) = dataset.split_with_ratio(0.9);

            println!(
                "Fit Multinomial Logistic Regression classifier with #{} training points",
                train.nsamples()
            );
            let model = train_model(&train);
            save_model(&model, cli.model_path.to_str().unwrap());

            let pred = model.predict(&valid);
            let cm = pred.confusion_matrix(&valid).unwrap();

            // // Print the confusion matrix, this will print a table with four entries. On the diagonal are
            // // the number of true-positive and true-negative predictions, off the diagonal are
            // // false-positive and false-negative
            println!("{:?}", cm);

            // // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
            // // predicted and targets)
            println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
        }
        Commands::Predict { json_value } => {
            let model = load_model(cli.model_path.to_str().unwrap());
            let features_ndarray = json_to_ndarray(json_value).unwrap();
            let pred = model.predict(&features_ndarray);
            println!("{:?}", pred);
        }
        Commands::Serve { bind_address } => {
            match serve(cli.model_path.to_str().unwrap(), bind_address) {
                Ok(_) => println!("Server started"),
                Err(e) => println!("Error starting server: {}", e),
            }
        }
    }
}
