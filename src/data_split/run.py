#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download inputs artifact
    logger.info(f"Fetching Dataset {args.ts_input}")
    artifact_local_path = run.use_artifact(args.ts_input).file()
    logger.info('Loading Time Series Dataset.')
    ts = pd.read_csv(artifact_local_path, decimal=",",parse_dates=["date"], index_col="date")
    ts = ts.apply(pd.to_numeric)
    ts.index = pd.to_datetime(ts.index)

    # Download inputs artifact
    logger.info(f"Fetching Dataset {args.tx_input}")
    artifact_local_path = run.use_artifact(args.tx_input).file()
    logger.info('Loading Tabular Dataset.')
    tx = pd.read_csv(artifact_local_path, decimal=",",parse_dates=["date"], index_col="date")
    tx = tx.apply(pd.to_numeric)


    # Splitting the Tabular Data
    logger.info("Splitting train/val and test (Tabular Dataset)")
    trainval, test = train_test_split(
        tx,
        test_size=args.test_size,
        random_state=args.random_seed
    )

    # Save to output files
    for df, k in zip([trainval, test], ['train', 'test']):
        logger.info(f"Uploading {k}_tx.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:

            df.to_csv(fp.name, index=False)

            log_artifact(
                f"{k}_tx.csv",
                f"{k}_tx",
                f"{k} split of the tabular dataset",
                fp.name,
                run,
            )

    # Splitting the Time Series Data
    logger.info("Splitting train/val and test (Time Series Dataset)")
    trainval = ts[:args.ts_test_size]
    test     = ts[args.ts_test_size:]
    # Save to output files
    for df, k in zip([trainval, test], ['train', 'test']):
        logger.info(f"Uploading {k}_ts.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:

            df.to_csv(fp.name, index=False)

            log_artifact(
                f"{k}_ts.csv",
                f"{k}_ts",
                f"{k} split of the time series dataset",
                fp.name,
                run,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument(
        "--ts_input", type=str, help="Time Series Data to split (a CSV file)", required=True
    )
    
    parser.add_argument(
        "--tx_input", type=str, help="Tabular Data to split (a CSV file)", required=True
    )

    parser.add_argument(
        "--test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items", required=True
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=True
    )

    parser.add_argument(
        "--ts_test_size", type=str, help="Size of the test split. Date from the test should be used", default='2017-08', required=True
    )

    args = parser.parse_args()

    go(args)
