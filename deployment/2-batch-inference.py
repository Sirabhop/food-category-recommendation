# Databricks notebook source
from preprocessing.load_data import clean_sentence, load_data
from evaluation.evaluation import get_label_and_probability

from databricks.feature_store import FeatureStoreClient

from mlflow.tensorflow import load_model
from datetime import datetime, timedelta
from pyspark.sql.types import StringType
from pyspark.sql import functions as F

import numpy as np
import pandas as pd
import tensorflow_text
import mlflow

# COMMAND ----------

dbutils.widgets.text('CATALOG_NAME', 'playground_prod')
dbutils.widgets.text('SCHEMA_NAME', 'ml_dish_tagging')
dbutils.widgets.text("MODEL_NAME", "dish_tagging_model")
dbutils.widgets.text("TRAINING_DATA_TABLE_NAME", "training_data")
dbutils.widgets.text("RANDOM_SEED", "112")
dbutils.widgets.text("FINAL_TABLE_NAME", "dish_tagging")

# Set up variable
list_variable = ['CATALOG_NAME', 'SCHEMA_NAME', 'MODEL_NAME', 'TRAINING_DATA_TABLE_NAME', 'RANDOM_SEED', 
                 'FINAL_TABLE_NAME']
dict_variable = {}
for variable in list_variable:
    dict_variable[variable] = dbutils.widgets.get(variable)

if dict_variable['CATALOG_NAME'] == 'playground_prod':
    FEATURE_PATH = 'prod.sv_aurora_major_merchant_vw.menu_a_d'
else:
    FEATURE_PATH = 'nonprod.sv_aurora_major_merchant_vw.menu_a_d'

DISHTAGGING_MODEL_PATH = '.'.join([dict_variable['CATALOG_NAME'], dict_variable['SCHEMA_NAME'], dict_variable['MODEL_NAME']])
FILE_PATH = '.'.join([dict_variable['CATALOG_NAME'], dict_variable['SCHEMA_NAME'], dict_variable['FINAL_TABLE_NAME']])
TRAINING_PATH = '.'.join([dict_variable['CATALOG_NAME'], dict_variable['SCHEMA_NAME'], dict_variable['TRAINING_DATA_TABLE_NAME']])

# COMMAND ----------

# Check if new menu has been appended?
count_menu = spark.sql(f"""
                        SELECT menu_id, menu_name FROM {FEATURE_PATH}
                        """).count()

if count_menu > 0:
    df_new_menu = spark.sql(f"""
                            SELECT menu_id, menu_name FROM {FEATURE_PATH}
                            WHERE menu_id NOT IN (SELECT menu_id FROM {FILE_PATH})
                            """).toPandas()

    if df_new_menu.shape[0] == 0:
        dbutils.notebook.exit('No new menu has been created, abort the batch inference script.')
    else:
        print(f'There is {df_new_menu.shape[0]} menu to label')
else:
    dbutils.notebook.exit(f'Got 0 record inside {FEATURE_PATH}, connect with DE in this regard.')

# COMMAND ----------

#Clean menu name
df_new_menu['cleaned_menu_name'] = df_new_menu['menu_name'].apply(clean_sentence)

# COMMAND ----------

#Load model
mlflow.set_registry_uri('databricks-uc')
model = load_model(f'models:/{DISHTAGGING_MODEL_PATH}@champion')

# COMMAND ----------

# Model prediction
result = model.predict(df_new_menu.cleaned_menu_name.values, batch_size=1024)
interpreted_result = get_label_and_probability(result)

# COMMAND ----------

# Get label and prob
df_new_menu['category_id'] = [label[0] for label in interpreted_result]
df_new_menu['prob'] = [label[1] for label in interpreted_result]

# COMMAND ----------

# Inverse encoded label
dish_type_encoder = spark.sql("SELECT * FROM playground_prod.ml_dish_tagging.dish_type_encoder") \
    .toPandas() \
    .set_index('category_id') \
    .to_dict()['dish_type_l3']

# COMMAND ----------

df_new_menu['dish_type_l3'] = df_new_menu.category_id.apply(lambda x: dish_type_encoder[x] if x != 0 else 'other')

# COMMAND ----------

# Inverse encoded label
dish_type_encoder = spark.sql("SELECT * FROM playground_prod.ml_dish_tagging.dish_type_encoder") \
    .toPandas() \
    .set_index('category_id') \
    .to_dict()['dish_type_l3']

# COMMAND ----------

# Prepare new dish type to replace
cat_name_mapper = spark.table('playground_prod.ml_dish_tagging.new_category_name_mapper').toPandas()
cat_name_mapper['new_dish_type'] = cat_name_mapper['new_dish_type'].str.lower()
cat_name_mapper['new_dish_type'] = cat_name_mapper['new_dish_type'].str.replace(' ', '_')

dict_cat_name_mapper = cat_name_mapper.set_index('dish_type_l3').to_dict()['new_dish_type']

def map_old_with_new_category_name(v):
    if v in dict_cat_name_mapper.keys():
        return dict_cat_name_mapper[v]
    else:
        return v

df_new_menu['new_dish_type_l3'] = df_new_menu['dish_type_l3'].apply(lambda x:  map_old_with_new_category_name(x))

# COMMAND ----------

# Prepare new category id
new_cat_id_enc = spark.table('playground_prod.ml_dish_tagging.new_category_id_mapper').toPandas()
new_cat_id_enc.columns = ['new_dish_type_l3', 'new_category_id']

df_new_menu = df_new_menu.merge(new_cat_id_enc, on = 'new_dish_type_l3', how = 'left')

# COMMAND ----------

int_cat_id = ['new_category_id', 'category_id']
for cat in int_cat_id:
    df_new_menu[cat] = df_new_menu[cat].astype(int)

# COMMAND ----------

# Set date
df_new_menu['run_date'] = datetime.now().date()

# COMMAND ----------

del df_new_menu['cleaned_menu_name']

# COMMAND ----------

ds = spark.createDataFrame(df_new_menu)

# COMMAND ----------

ds.write.mode('append').saveAsTable(FILE_PATH)