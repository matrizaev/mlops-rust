use std::path::PathBuf;

use hf_hub::api::sync::ApiError;
use linfa::prelude::*;
use linfa_logistic::{MultiFittedLogisticRegression, MultiLogisticRegression};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use polars::prelude::*;

type CustomDatasetType = DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>>,
>;

type CustomTrainedModel = MultiFittedLogisticRegression<f64, String>;

pub fn download_dataset(repo: &str, name: &str) -> Result<PathBuf, ApiError> {
    let api = hf_hub::api::sync::Api::new().unwrap();
    api.dataset(String::from(repo)).get(name)
}

pub fn read_dataset(path: &str, feature_names: &[&str], target_name: &str) -> CustomDatasetType {
    let df = CsvReader::from_path(path).unwrap().finish().unwrap();

    let x_train = df
        .select(feature_names)
        .unwrap()
        .clone()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();

    let y_train: Vec<String> = df
        .column(target_name)
        .unwrap()
        .clone()
        .str()
        .unwrap()
        .into_iter()
        .filter_map(|opt_str| opt_str.map(|s| s.to_string()))
        .collect();

    Dataset::new(x_train, y_train.into()).with_feature_names(feature_names.to_vec())
}

pub fn train_model(dataset: &CustomDatasetType) -> CustomTrainedModel {
    // fit a Logistic regression model with 150 max iterations
    MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(dataset)
        .unwrap()
}

pub fn save_model(model: &CustomTrainedModel, path: &str) {
    let serialized = serde_pickle::to_vec(model, Default::default()).unwrap();
    std::fs::write(path, serialized).unwrap();
}

pub fn load_model(path: &str) -> CustomTrainedModel {
    let serialized = std::fs::read(path).unwrap();
    serde_pickle::from_slice(&serialized, Default::default()).unwrap()
}
