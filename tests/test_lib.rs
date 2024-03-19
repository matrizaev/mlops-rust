use linfa::dataset::Records;
use linfa::prelude::Predict;
use mlops_rust::model::{download_dataset, read_dataset, train_model, FEATURE_NAMES};
use ndarray::arr1;

#[test]
fn test_download_dataset() {
    let result = download_dataset("test", "test");
    assert!(result.is_err());
    let result = download_dataset("scikit-learn/iris", "Iris.csv");
    assert!(result.is_ok());
}

#[test]
fn test_read_dataset() {
    let result = read_dataset("tests/test.csv");
    assert!(result.nsamples() == 2);
    assert!(result.nfeatures() == 4);
    assert!(result.ntargets() == 1);
    assert!(result.feature_names() == FEATURE_NAMES);
}

#[test]
fn test_train_model() {
    let dataset = read_dataset("tests/test.csv");
    let model = train_model(&dataset).unwrap();
    let pred = model.predict(&dataset);
    assert_eq!(pred, arr1(&["test", "protest"]));
}
