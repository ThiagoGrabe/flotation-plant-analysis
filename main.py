import json
import logging

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "get_data",
    "basic_cleaning",
    "create_datasets",
    "data_split",
    "tx_pepiline"
]

_models = [
    "lasso",
    "xgboost",
    "random_forest"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Models to train/execute
    # Steps to execute
    models_par = config['main']['models']
    active_models = models_par.split(",") if models_par != "all" else _models

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "get_data"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "MiningProcess_Flotation_Plant_Database.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
           _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "MiningProcess_Flotation_Plant_Database.csv:latest",
                    "output_artifact": "cleansed_mining_flotation_plant.csv",
                    "output_type": "cleansed_mining_flotation_plant",
                    "output_description": "Cleansed data."
                },
            )
           
        if "create_datasets" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "create_datasets"),
                "main",
                parameters={
                    "input_artifact": "cleansed_mining_flotation_plant.csv:latest",
                    "ts": config["modeling"]["ts"],
                    "tx": config["modeling"]["tx"],
                    "output_artifact_ts": "ts.csv",
                    "output_type_ts": "time_series_data",
                    "output_description_ts": "Time Series data.",
                    "output_artifact_tx": "tx.csv",
                    "output_type_tx": "tabular_data",
                    "output_description_tx": "tabular data."
                },
            )

        if "data_check" in active_steps:
            raise NotImplementedError
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "ref": "cleansed_mining_flotation_plant.csv:reference",
                    "csv": "cleansed_mining_flotation_plant.csv:latest",
                    "kl_threshold" : config["data_check"]["kl_threshold"]
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_split"),
                "main",
                parameters={
                    "ts_input": "ts.csv:latest",
                    "tx_input": "tx.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "ts_test_size": config["modeling"]["ts_test_size"]
                    }
                )

        if "tx_pipeline" in active_steps:

            # NOTE: we need to serialize the model configuration into JSON

            configs = ['lasso_config.json', 'rf_config.json', 'xg_config.json']
            names   = ['lasso', 'random_forest', 'xgboost']

            for json_config, names_config in zip(configs, names):
                
                if names_config in active_models:
                    pass
                else:
                    continue

                _config = os.path.abspath(json_config)
                with open(_config, "w+") as fp:
                    json.dump(dict(config["modeling"][names_config].items()), fp)

                logger.info(f"The configuration for model {names_config} is {_config}")

                _ = mlflow.run(
                    os.path.join(hydra.utils.get_original_cwd(), "src", "tx_pipeline"),
                    "main",
                    parameters={
                        "trainval_artifact": "train_tx.csv:latest",
                        "val_size": config['modeling']['val_size'],
                        "random_seed": config['modeling']['random_seed'],
                        "model_config": _config,
                        "model_name"  : names_config,
                        "folds" :config['modeling']['kfolds'],
                        "output_artifact" : f"{names_config}_export"
                        }
                    )

        if "test_regression_model" in active_steps:
             _ = mlflow.run(
                 os.path.join(hydra.utils.get_original_cwd(), "src", "test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                    }
                )


if __name__ == "__main__":
    go()
