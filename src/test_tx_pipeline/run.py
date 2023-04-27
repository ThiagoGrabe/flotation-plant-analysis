#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from importlib import import_module

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path, decimal=",").dropna()
    y_test = X_test.pop(args.target_variable)
    logger.info(f"Target Feature {args.target_variable}")
    run.log({"Target Feature" : args.target_variable})

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Metrics on Testing Set\n")
    for metric_name in ['r2_score','mean_squared_error','mean_absolute_percentage_error','mean_absolute_error']:
        module = import_module('sklearn.metrics')
        metric = getattr(module,metric_name)
        if metric_name == 'mean_squared_error':
            metric_value = metric(y_test, y_pred, squared=False)
            logger.info(f"Validation {metric_name}: {round(metric_value,4)}")
            run.log({f"Validation {metric_name}": round(metric_value,4)})
        else:
            metric_value = metric(y_test, y_pred)
            logger.info(f"Validation {metric_name}: {round(metric_value,4)}")
            run.log({f"Validation {metric_name}": round(metric_value,4)})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str, 
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    parser.add_argument(
        "--target_variable",
        type=str, 
        help="Target Variable to be predicted. Must be a string that matches with the dataframe.",
        required=True
    )

    args = parser.parse_args()

    go(args)
