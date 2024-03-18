use std::path::PathBuf;

use hf_hub::api::sync::ApiError;
use linfa::prelude::*;
use linfa_logistic::MultiLogisticRegression;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use polars::prelude::*;
// use rand::*;

fn download_dataset(repo: &str, name: &str) -> Result<PathBuf, ApiError> {
    let api = hf_hub::api::sync::Api::new().unwrap();
    api.dataset(String::from(repo)).get(name)
}

fn read_dataset(
    path: &str,
    feature_names: &[&str],
    target_name: &str,
) -> DatasetBase<
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<String>, Dim<[usize; 1]>>,
> {
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

pub fn main() {
    let path = download_dataset("scikit-learn/iris", "Iris.csv").unwrap();

    let feature_names = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ];

    let target_name = "Species";

    let dataset = read_dataset(path.to_str().unwrap(), &feature_names, target_name);

    // shuffle the dataset
    // let mut rng = thread_rng();
    // dataset.shuffle(&mut rng);

    let (train, valid) = dataset.split_with_ratio(0.9);

    println!(
        "Fit Multinomial Logistic Regression classifier with #{} training points",
        train.nsamples()
    );

    // fit a Logistic regression model with 150 max iterations
    let model = MultiLogisticRegression::default()
        .max_iterations(50)
        .fit(&train)
        .unwrap();

    // // predict and map targets
    let pred = model.predict(&valid);

    // // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // // the number of true-positive and true-negative predictions, off the diagonal are
    // // false-positive and false-negative
    println!("{:?}", cm);

    // // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
}
