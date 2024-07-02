# Databricks notebook source
DATA_ENV='prod'
DEV_ENV='pd'

global DATA_ENV, DEV_ENV

dbutils.widgets.text('DYNAMODB_SCOPE', f'ppv-dataplatform-{DATA_ENV}-dynamodb-read-scope')
dbutils.widgets.text('DYNAMODB_ACCESS_KEY', f'ppv-dataplatform-{DATA_ENV}-dynamodb-read-access-key-id')
dbutils.widgets.text('DYNAMODB_SECRET_KEY', f'ppv-dataplatform-{DATA_ENV}-dynamodb-read-secret-access-key')
dbutils.widgets.text('EXPERIMENT_ID', '3826439196377487')
dbutils.widgets.text('MODEL_NAME', f'food-dbb-{DEV_ENV}')
dbutils.widgets.text('CATALOG', 'prod')
dbutils.widgets.text('ENDPOINT_NAME', 'prod')

list_variable = ['DYNAMODB_SCOPE', 'DYNAMODB_ACCESS_KEY', 'DYNAMODB_SECRET_KEY', 'EXPERIMENT_ID', 'MODEL_NAME', 'CATALOG']
dict_variable = {}
for variable in list_variable:
    dict_variable[variable] = dbutils.widgets.get(variable)

FEATURE_TABLE = f"{dict_variable['CATALOG']}.gd_food.food_customer_category_list_a_d"
global FEATURE_TABLE

# COMMAND ----------

import logging
import boto3
import os
import json
import mlflow
import datetime
from boto3.dynamodb.conditions import Key, Attr

from mlflow import start_run, register_model, last_active_run, MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import log_model, load_model, PythonModel
from mlflow.deployments import get_deploy_client

# COMMAND ----------

class getPersonalizedDishes(PythonModel):

    def __init__(self, keys):
        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.aws_access_key = keys[0]
        self.aws_secret_key = keys[1]
        self.dynamoDB_table_name = FEATURE_TABLE

    def load_context(self, context):
        import json

        self.client = boto3.client('dynamodb', 
                            region_name='ap-southeast-1',
                            aws_access_key_id=self.aws_access_key,
                            aws_secret_access_key=self.aws_secret_key
                            )

    def __query_from_dynamo(self, user_id):
        response = self.client.get_item(
            TableName=self.dynamoDB_table_name,
            Key={
                '_feature_store_internal__primary_keys': {
                    'S': f'["{user_id}"]'
                }
            },
            AttributesToGet=[
                'gd_dish_types'
            ]
        )
        return response
    
    def __format_output(self, items):
        format_res = json.loads(items)
        return  format_res

    def __postprocess_result(self, user_id, response):
        if "Item" in response:
            res = response['Item']['gd_dish_types']['S']
            format_output = self.__format_output(res)
            return {'user_id': user_id, 'recommended_items': format_output}
        else:
            query_items = self.__query_from_dynamo('default')
            res = query_items['Item']['gd_dish_types']['S']
            format_output = self.__format_output(res)
            return {'user_id': user_id, 'recommended_items': format_output}

    def predict(self, context, model_input):
        user_id = model_input['user_id'].values[0]
        query_items = self.__query_from_dynamo(user_id)
        result = self.__postprocess_result(user_id, query_items)
        return result

# COMMAND ----------

# Wrap model
DYNAMODB_ACCESS_KEY = dbutils.secrets.get(dict_variable['DYNAMODB_SCOPE'], dict_variable['DYNAMODB_ACCESS_KEY'])
DYNAMODB_SECRET_KEY = dbutils.secrets.get(dict_variable['DYNAMODB_SCOPE'], dict_variable['DYNAMODB_SECRET_KEY'])

wrapped_model = getPersonalizedDishes(keys=[DYNAMODB_ACCESS_KEY, DYNAMODB_SECRET_KEY])

# COMMAND ----------

# Define Pyfunc model signature
input_schema = Schema([ColSpec("string", 'user_id')])
signature = ModelSignature(inputs=input_schema)

# COMMAND ----------

# Define input example
input_example = {
    'user_id': 'default'
}

# COMMAND ----------

# Set experiment
with start_run(experiment_id=dict_variable['EXPERIMENT_ID']):
    log_model("model", 
              python_model=wrapped_model, 
              input_example=input_example, 
              signature=signature
              )

# COMMAND ----------

last_run_info =last_active_run()
run_id = last_run_info.info.run_id
run_id

# COMMAND ----------

loaded_model = load_model(f"runs:/{run_id}/model")

# COMMAND ----------

predicted_result = loaded_model.predict({'user_id':'default'})
predicted_result

# COMMAND ----------

client = MlflowClient()

client.log_param(
    run_id=run_id,
    key="default_list",
    value=predicted_result
)

# COMMAND ----------

# Upload model
result=register_model(f"runs:/{run_id}/model", dict_variable['MODEL_NAME'])

# COMMAND ----------

def get_latest_model_version(model_name, stage=None):
    """Fetches the latest version of the specified registered MLflow model.

    Args:
        model_name (str): The name of the registered model.
        stage (str, optional): The stage to filter by (e.g., "Staging", "Production", "None"). Defaults to None (latest overall version).

    Returns:
        mlflow.entities.model_registry.ModelVersion: The latest model version object, or None if no model is found.
    """
    client = MlflowClient()

    # Get all versions of the model
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error retrieving model versions: {e}")
        return None
    
    # Check if no versions exist
    if not versions:
        print(f"No versions found for model: {model_name}")
        return None

    versions = [int(v.version) for v in versions]

    # Sort by version number (descending)
    latest_version = sorted(versions, reverse=True)[0]

    print(f"Latest version of model '{model_name}': {latest_version}")
    return latest_version

API_VERSION = get_latest_model_version(dict_variable['MODEL_NAME'])

# COMMAND ----------

CONFIG = {
  "served_entities": [
    {
    "name": dict_variable['MODEL_NAME'],
    "entity_name": dict_variable['MODEL_NAME'],
    "entity_version": str(API_VERSION),
    "workload_size": "Large",
    "scale_to_zero_enabled": False
    }
  ]
}

# COMMAND ----------

api_client = get_deploy_client("databricks")

endpoint = api_client.update_endpoint(
    endpoint=dict_variable['MODEL_NAME'],
    config=CONFIG)