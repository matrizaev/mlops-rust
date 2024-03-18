use linfa::prelude::*;

// use rand::*;

use mlops_rust::{download_dataset, read_dataset, train_model};

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

    let model = train_model(&train);

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
