#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info('Creating Basic Cleaning Artifacts')

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    data = pd.read_csv(artifact_local_path, decimal=",",parse_dates=["date"], index_col="date")
    df = data.copy()
    logger.info('Dataset loaded.')

    ###### Data Cleaning ######

    # As we have problatic dates in March, we start ou analysis from April.
    df = df.loc["2017-04-01":]

    # Slicing the data and getting the data from April 10th
    new_val = df.loc['2017-04-10 00:00:00'].values[-1]
    append_val = pd.DataFrame(new_val.reshape(1,-1), columns=df.columns.tolist())
    append_val['date'] = '2017-04-10 00:00:00'
    append_val['date'] = pd.to_datetime(append_val['date'])
    append_val = append_val.set_index('date', drop=True)
    df = pd.concat([df, append_val]).reset_index().set_index('date', drop=True)
    df = df.sort_index()

    # There are some interpolated values for % Silica and % Iron. Instead of using the interpolated data, we take the median of the values 
    # for an hour with the interpolated values.
    for idx in df.index.unique():
        silica_values = df.loc[idx]['% Silica Concentrate'].unique()
        iron_values   = df.loc[idx]['% Iron Concentrate'].unique()
        if len(silica_values) > 1:
            df.loc[idx, "% Silica Concentrate"] = np.median(df.loc[idx]["% Silica Concentrate"].values)
        if len(iron_values) > 1:
            df.loc[idx, "% Iron Concentrate"] = np.median(df.loc[idx]["% Iron Concentrate"].values)

    df.index = pd.date_range(start='2017-04-01 00:00:00', end='2017-09-09 23:59:40', freq='20s')
    df.index.names = ['date']
    logger.info('Dataset cleaned.')

    logger.info('Starting Basic Feature Engineering.')
    # Basic Feature Engineering
    df['Shift'] = np.nan
    for idx in df.between_time(start_time='00:00:00', end_time='08:00:00').index.tolist():
        df.loc[idx, "Shift"] = "A"
    for idx in df.between_time(start_time='08:01:00', end_time='16:00:00').index.tolist():
        df.loc[idx, "Shift"] = "B"
    for idx in df.between_time(start_time='16:01:00', end_time='23:59:59').index.tolist():
        df.loc[idx, "Shift"] = "C"
    logger.info('Finished Basic Feature Engineering.')
    
    logger.info('Saving Dataset.')
    # Saving to CSV
    df.to_csv(args.output_artifact, index=True)
    logger.info('Dataset saved.')

    # Uploading to W&B
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
    )
    artifact.add_file("cleansed_mining_flotation_plant.csv")
    run.log_artifact(artifact)
    logger.info('Logged Artifact.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Cleaning dataset",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the artifact.",
        required=True
    )

    args = parser.parse_args()

    go(args)
