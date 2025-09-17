# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.basic_model import BasicModel

from dotenv import load_dotenv
from mlops_course.utils import is_databricks
import os

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "1234567890abcd", "branch": "week2"})
print(config)
# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model_classifier()

# COMMAND ----------


# Search for all runs in the specified experiment and filter by tag.
# Do not add `[0]` to the end of this line.
runs_df = mlflow.search_runs(
    experiment_names=["/Shared/loan-default-basic"], 
    filter_string="tags.branch='week2'",
    # Sort the runs by the desired metric (F1-score) in descending order.
    # The name of the metric here must match exactly what you logged.
    order_by=["metrics.f1_score DESC"]
)

# Check if any runs were found
if not runs_df.empty:
    # Now that the DataFrame is sorted, the first row is the best run.
    best_run_id = runs_df.iloc[0].run_id
    print(f"The run_id of the best model (by F1-score) is: {best_run_id}")
    
    # You can also access other information about the best run
    best_run_info = runs_df.iloc[0]
    print("Information about the best run:")
    print(best_run_info)
    
else:
    print("No runs found with the specified criteria.")

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/loan-default-basic"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-classifier-model-loan-default")

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)
# COMMAND ----------