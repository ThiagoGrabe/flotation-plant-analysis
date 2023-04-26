#!/usr/bin/env python
"""
Create Datasets for Time Series Forecasting and Machine Learning Prediction given the Flotation Plant Task
"""
import argparse
import logging
import wandb

import pandas as pd
import numpy as np

from sktime.transformations.series.summarize import WindowSummarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="create_datasets")
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info('Loading the Dataset.')
    data = pd.read_csv(artifact_local_path, decimal=",",parse_dates=["date"], index_col="date")
    data = data.iloc[:,0:-2].apply(pd.to_numeric)
    
    df = data.copy()
    logger.info('Dataset loaded.')

    if args.ts:
        logger.info('Creating the Time Series Data.')
        kwargs = {
        "lag_feature": {
            "lag": [-180, -360, -540, -720],
            }
        }
        transformer = WindowSummarizer(**kwargs, target_cols=['% Silica Concentrate','% Iron Concentrate'])
        ts_transformed = transformer.fit_transform(df)
        logger.info('Time Series Data Transformed using Time Series Transformations.')

        # Saving to CSV
        ts_transformed.to_csv(args.output_artifact_ts, index=True)
        logger.info('Time Series Dataset saved.')

        # Uploading to W&B
        artifact = wandb.Artifact(
        args.output_artifact_ts,
        type=args.output_type_ts,
        description=args.output_description_ts,
        )

        artifact.add_file(f"{args.output_artifact_ts}")
        run.log_artifact(artifact)
        logger.info('Logged Time Series Artifact.')

    if args.tx:
        logger.info('Creating the Tabular Data.')
        resample_dict = dict(zip(ts_transformed.select_dtypes(include='number').columns.tolist(), ['mean']*(len(df)-2)))
        tx_transformed = ts_transformed.groupby([pd.Grouper(freq='h')]).agg(resample_dict)
        logger.info('Tabular Data Transformed using one hour sampling and the mean of each column for that period.')
        
        # Saving to CSV
        tx_transformed.to_csv(args.output_artifact_tx, index=True)
        logger.info('Tabular Dataset saved.')
        
        # Uploading to W&B
        artifact = wandb.Artifact(
        args.output_artifact_tx,
        type=args.output_type_tx,
        description=args.output_description_tx,
        )

        artifact.add_file(f"{args.output_artifact_tx}")
        run.log_artifact(artifact)
        logger.info('Logged Tabular Artifact.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create Datasets for Time Series Forecasting and Machine Learning Prediction given the Flotation Plant Task")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--ts", 
        type=bool,
        help="If ts is True, then a time series dataset will be created.",
        required=True
        )

    parser.add_argument(
        "--tx", 
        type=bool,
        help="If ts is True, then a tabular dataset will be created.",
        required=True
        )
    
    parser.add_argument(
        "--output_artifact_ts", 
        type=str,
        help="Output artifact for Time Series Data as csv",
        required=True
    )

    parser.add_argument(
        "--output_type_ts", 
        type=str,
        help="Time Series Data",
        required=True
    )

    parser.add_argument(
        "--output_description_ts", 
        type=str,
        help="Description of the Time Series Data.",
        required=True
    )

    parser.add_argument(
        "--output_artifact_tx", 
        type=str,
        help="Output artifact Tabular Data as csv",
        required=True
    )

    parser.add_argument(
        "--output_type_tx", 
        type=str,
        help="Tabular Data",
        required=True
    )

    parser.add_argument(
        "--output_description_tx", 
        type=str,
        help="Description of the Tabular Data.",
        required=True
    )
    
    args = parser.parse_args()

    go(args)
