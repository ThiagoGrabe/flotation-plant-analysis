#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
# import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from importlib import import_module

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_tx_pepiline")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.model_config) as fp:
        model_config = json.load(fp)
    run.config.update(model_config)

    # Fix the random seed for the model, if exists
    logger.info(f"Updating model {args.model_name}, if exists.")
    try:
        model_config['random_state'] = args.random_seed
    except:
        logger.info(f"Model {args.model_name} does not have a random state to be updated.")


    ######################################
    # Use run.use_artifact(...).file() to get the train and validation artifact (args.trainval_artifact)
    # and save the returned path in train_local_pat
    logger.info(f"Fetching the train artifact {args.trainval_artifact}")
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    ######################################
    X = pd.read_csv(trainval_local_path, decimal=",")
    X = X.apply(pd.to_numeric).dropna()

    logger.info(f"Target Feature {args.target_variable}")
    run.log({"Target Feature" : args.target_variable})
    y = X.pop(args.target_variable)  # this removes the target column from X and puts it into y

    logger.info(f"Datasets X and y are ready to be used during training.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.random_seed
    )

    cv = KFold(n_splits=args.folds)
    
    logger.info("Preparing Sklearn Pipeline - Training and Fitting")

    sk_pipe = get_inference_pipeline(model_config)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting the data...")

    ######################################
    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    scores = cross_validate(sk_pipe, 
                            X_train, 
                            y_train, 
                            n_jobs=-1, 
                            scoring=['neg_mean_squared_error', 
                                     'neg_mean_absolute_percentage_error', 
                                     'neg_mean_absolute_error', 
                                     'r2'], 
                                     cv=cv, 
                                     return_train_score=True, 
                                     return_estimator=True
                                     )
    sk_pipe.fit(X_train, y_train)
    ######################################
    logger.info("Validation Scores")
    for test_score in [k for k in list(scores.keys()) if 'test_' in k]:
        if test_score == 'r2':
            logger.info(f"Metric: {test_score}\n--Mean: {np.mean(scores[test_score])} +- {np.std(scores[test_score])}")

            run.summary[f'{test_score}'] = np.mean(scores[test_score])
            run.log({f'{test_score}': np.mean(scores[test_score])})
        else:
            logger.info(f"Metric: {test_score}\n--Mean: {np.mean(scores[test_score])} +- {np.std(scores[test_score])}")

            run.summary[f'{test_score}'] = np.mean(scores[test_score]*-1)
            run.log({f'{test_score}': np.mean(scores[test_score]*-1)})
    ######################################
    logger.info("Training Scores")
    for test_score in [k for k in list(scores.keys()) if 'train_' in k]:
        if test_score == 'r2':
            logger.info(f"Metric: {test_score}\n--Mean: {round(np.mean(scores[test_score]),4)} +- {round(np.std(scores[test_score]),4)}")

            run.summary[f'{test_score}'] = np.mean(scores[test_score])
            run.log({f'{test_score}': np.mean(scores[test_score])})
        else:
            logger.info(f"Metric: {test_score}\n--Mean: {round(np.mean(scores[test_score]),4)} +- {round(np.std(scores[test_score]),4)}")

            run.summary[f'{test_score}'] = round(np.mean(scores[test_score]*-1),4)
            run.log({f'{test_score}': round(np.mean(scores[test_score]*-1),4)})
    ######################################
    logger.info("Metrics for Validation Set\n")
    y_pred = sk_pipe.predict(X_val)
    for metric_name in ['r2_score','mean_squared_error','mean_absolute_percentage_error','mean_absolute_error']:
        module = import_module('sklearn.metrics')
        metric = getattr(module,metric_name)
        if metric_name == 'mean_squared_error':
            metric_value = metric(y_val, y_pred, squared=False)
            logger.info(f"Validation {metric_name}: {round(metric_value,4)}")
            run.log({f"Validation {metric_name}": round(metric_value,4)})
        else:
            metric_value = metric(y_val, y_pred)
            logger.info(f"Validation {metric_name}: {round(metric_value,4)}")
            run.log({f"Validation {metric_name}": round(metric_value,4)})

    logger.info("Exporting model...")

    # Save model package in the MLFlow sklearn format
    if os.path.exists(f"{args.model_name}_dir"):
        shutil.rmtree(f"{args.model_name}_dir")

    ######################################
    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory f"{args.model_name}_dir"
    mlflow.sklearn.save_model(sk_pipe, f"{args.model_name}_dir")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export",
        description=f"Uploading {args.model_name} model using MLFlow.",
        metadata=model_config,
    )
    artifact.add_dir(f"{args.model_name}_dir")
    logger.info("Uploading model to Weight&Biases")
    run.log_artifact(artifact)
   

    # Upload to W&B the feture importance visualization
    # run.log(
    #     {
    #       "feature_importance": wandb.Image(fig_feat_imp),
    #     }
    # )




def get_inference_pipeline(model_config):
    
    module = import_module(model_config["module"])
    model = getattr(module,model_config["estimator"])

    hyperparameters = dict()
    for param in list(filter(lambda x: x in list(model().get_params().keys()), list(model_config.keys()))):
        hyperparameters.update({param : model_config[param]})

    sk_pipe = Pipeline(
            [   
                ('preprocess', StandardScaler()), 
                ('model', model().set_params(**hyperparameters))
            ]
                    )

    return sk_pipe


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--target_variable",
        type=str, 
        help="Target Variable to be predicted. Must be a string that matches with the dataframe.",
        required=True
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--model_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--model_name",
        help="Model name",
    )

    parser.add_argument(
        "--folds",
        type=int,
        help="Folds to used in the K-Fold Cross Validation for Tabular Machine Learning application.",
        default=5,
        required=False,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
