# Databricks notebook source
# Library
from preprocessing.load_data import load_data
from model.build_model import build_model

# Logging lib
from databricks.feature_store import FeatureStoreClient

from mlflow import MlflowClient, last_active_run, set_registry_uri, register_model, set_experiment
from mlflow.tensorflow import autolog, load_model

# COMMAND ----------

# Set up variable
list_variable = ['CATALOG_NAME', 'SCHEMA_NAME', 'MODEL_NAME', 'TRAINING_DATA_TABLE_NAME', 'RANDOM_SEED', 'FINAL_TABLE_NAME', 'EPOCH', 'EXPERIMENT_ID']
dict_variable = {}
for variable in list_variable:
    dict_variable[variable] = dbutils.widgets.get(variable)

DISHTAGGING_MODEL_PATH = '.'.join([dict_variable['CATALOG_NAME'], dict_variable['SCHEMA_NAME'], dict_variable['MODEL_NAME']])
FILE_PATH = '.'.join([dict_variable['CATALOG_NAME'], dict_variable['SCHEMA_NAME'], dict_variable['FINAL_TABLE_NAME']])
TRAINING_PATH = '.'.join([dict_variable['CATALOG_NAME'], dict_variable['SCHEMA_NAME'], dict_variable['TRAINING_DATA_TABLE_NAME']])

# COMMAND ----------

# Autolog for tensorflow
set_registry_uri('databricks-uc')
autolog(registered_model_name=DISHTAGGING_MODEL_PATH)
set_experiment(experiment_id=dict_variable['EXPERIMENT_ID'])

fs = FeatureStoreClient()
client = MlflowClient()

spark.sql(f"USE CATALOG {dict_variable['CATALOG_NAME']}")
spark.sql(f"USE SCHEMA {dict_variable['SCHEMA_NAME']}")

# COMMAND ----------

# Prepare a dataset
X_train, X_val, X_test, y_train, y_val, y_test, training_set, dish_type_list, label_encoder = load_data(fs, TRAINING_PATH)
print('Successfully prepared training and testing data')

# COMMAND ----------

# Prepare a model
model, callbacks = build_model(len(dish_type_list))
print('Start fine-tuning a model')

# COMMAND ----------

# Train a model
train_history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=int(dict_variable['EPOCH']), 
            callbacks=[callbacks],
            batch_size=16)
print('Complete fine-tuning')

# COMMAND ----------

# Get run_id and latest version
model_version_infos = client.search_model_versions("name = '%s'" % DISHTAGGING_MODEL_PATH)

max_version_run_id = max(model_version_infos, key=lambda mv: int(mv.version)).run_id
max_version = max(model_version_infos, key=lambda mv: int(mv.version)).version

# COMMAND ----------

# Log metrices
test_result = model.evaluate(X_test, y_test)

client.log_metric(run_id=max_version_run_id,
                  key="test_loss", value=test_result[0])
client.log_metric(run_id=max_version_run_id,
                  key="test_accuracy", value=test_result[1])

# COMMAND ----------

# Label the latest model
client.set_registered_model_alias(DISHTAGGING_MODEL_PATH, "champion", int(max_version))
client.set_registered_model_alias(DISHTAGGING_MODEL_PATH, "old", int(max_version)-1)