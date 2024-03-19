use std::error::Error;
use std::path::PathBuf;

use hf_hub::api::sync::ApiError;
use linfa::prelude::*;
use linfa_logistic::{MultiFittedLogisticRegression, MultiLogisticRegression};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use polars::prelude::*;
use std::io::Cursor;

use mlflow::{timestamp, Client};

type CustomDatasetType = DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>>,
>;

type CustomInferenceType = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;

pub type CustomTrainedModel = MultiFittedLogisticRegression<f64, String>;

pub const FEATURE_NAMES: [&str; 4] = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
];

pub const TARGET_NAME: &str = "Species";

pub fn download_dataset(repo: &str, name: &str) -> Result<PathBuf, ApiError> {
    let api = hf_hub::api::sync::Api::new().unwrap();
    api.dataset(String::from(repo)).get(name)
}

pub fn read_dataset(path: &str) -> CustomDatasetType {
    let df = CsvReader::from_path(path).unwrap().finish().unwrap();

    let x_train = df
        .select(FEATURE_NAMES)
        .unwrap()
        .clone()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();

    let y_train: Vec<String> = df
        .column(TARGET_NAME)
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .filter_map(|opt_str| opt_str.map(|s| s.to_string()))
        .collect();

    Dataset::new(x_train, y_train.into()).with_feature_names(FEATURE_NAMES.to_vec())
}

pub fn train_track_model(
    dataset: &CustomDatasetType,
    mlflow_tracking_uri: &str,
    mlflow_experiment_name: &str,
) -> Result<(), Box<dyn Error>> {
    println!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        dataset.nsamples()
    );

    let client = Client::for_server(mlflow_tracking_uri);

    let experiment = match client
        .get_experiment(mlflow_experiment_name)
        .or_else(|| client.create_experiment(mlflow_experiment_name))
    {
        Some(experiment) => experiment,
        None => panic!("Error: Could not create experiment."),
    };

    let run = experiment.create_run();

    let model = train_model(dataset)?;
    let pred = model.predict(dataset);
    let cm = pred.confusion_matrix(dataset)?;

    run.log_metric("accuraccy", cm.accuracy().into(), timestamp(), 0);

    run.terminate();
    Ok(())
}

pub fn train_model(dataset: &CustomDatasetType) -> Result<CustomTrainedModel, Box<dyn Error>> {
    // fit a Logistic regression model with 150 max iterations
    let model = MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(dataset)?;
    Ok(model)
}

pub fn save_model(model: &CustomTrainedModel, path: &str) {
    let serialized = serde_pickle::to_vec(model, Default::default()).unwrap();
    std::fs::write(path, serialized).unwrap();
}

pub fn load_model(path: &str) -> CustomTrainedModel {
    let serialized = std::fs::read(path).unwrap();
    serde_pickle::from_slice(&serialized, Default::default()).unwrap()
}

pub fn json_to_ndarray(json_value: &str) -> Result<CustomInferenceType, PolarsError> {
    let cursor = Cursor::new(json_value);
    JsonReader::new(cursor)
        .finish()?
        .select(FEATURE_NAMES)?
        .to_ndarray::<Float64Type>(IndexOrder::C)
}
