# Databricks notebook source
# MAGIC %md
# MAGIC *Note: This script crate food_customer_category_list_a_d, which is a feature table (+ online serving with DynamoDB)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField

from datetime import timedelta, datetime
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.feature_store import FeatureStoreClient
from databricks.feature_engineering.online_store_spec import AmazonDynamoDBSpec

busi_dt = str(datetime.today().date() - timedelta(1))
busi_dt

# COMMAND ----------

# Set up variable
dbutils.widgets.text('layer', 'gd')
dbutils.widgets.text('src_sys', 'food')
dbutils.widgets.text("target_db_nm", "gd_food")
dbutils.widgets.text("target_tbl_nm", "food_customer_category_list_a_d")
dbutils.widgets.text("catalog", "prod")
dbutils.widgets.text("busi_dt", busi_dt)

LAYER_NAME = dbutils.widgets.get('layer')
CATALOG_NAME = dbutils.widgets.get("catalog")
SRC_SYS = dbutils.widgets.get('src_sys')
BUSI_DT = dbutils.widgets.get("busi_dt")
TARGET_DB_NM = dbutils.widgets.get("target_db_nm")
TARGET_TBL_NM = dbutils.widgets.get("target_tbl_nm")

FEATURE_TABLE = f"{CATALOG_NAME}.{TARGET_DB_NM}.{TARGET_TBL_NM}"
FEATURE_TABLE

# COMMAND ----------

def convert_delta_to_feature(fe, delta_df, feature_table, primary_keys, description, mode='merge'):
    try:
        print("Check feature store is already existed or not")
        fs = FeatureStoreClient()
        fs.get_table(feature_table)
    except:
        print("Create new feature store table")
        fe.create_table(
        name=feature_table,
        primary_keys=primary_keys,
        df=delta_df,
        schema=delta_df.schema,
        description=description
    )
    print("Write data to feature store table")
    fe.write_table(
        name=feature_table,
        df = delta_df,
        mode = mode
    )

def drop_features_store_table(fe, feature_table):
        fe.drop_table(name=feature_table)

def publish_to_dynamodb(fe, catalog , dynamodb_table, mode='merge'):
    online_store = AmazonDynamoDBSpec(
        region='ap-southeast-1',
        read_secret_prefix=f'ppv-dataplatform-prod-dynamodb-read-scope/ppv-dataplatform-prod-dynamodb-read'
    )
    fe.publish_table(
        name=dynamodb_table,
        online_store=online_store,
        mode=mode
    )

def drop_dynamodb_table(fe,dynamodb_table):
    online_store = AmazonDynamoDBSpec(
        region='ap-southeast-1'
    )
    fe.drop_online_table(
        name=dynamodb_table,
        online_store = online_store
    )


# COMMAND ----------

df = spark.sql(
f"""
WITH RankedDishes AS (
  SELECT
    user_id,
    dish_type_l3,
    category_id,
    decile,
    rank,
    ROW_NUMBER() OVER (
      PARTITION BY user_id
      ORDER BY
        decile DESC,
        rank DESC
    ) AS row_num
  FROM
    (
      SELECT * FROM {CATALOG_NAME}.{TARGET_DB_NM}.food_customer_category_s_d
      WHERE dish_type_l3 NOT IN ('rice_noodle', 'soup', 'utensil', 'rice', 'hotpot', 'other', 'seasoning', 'chili_paste')
    )
)
SELECT
  user_id,
  dish_type_l3,
  CAST(category_id AS INT) AS category_id
FROM
  RankedDishes
WHERE
  row_num <= 10 
ORDER BY
  user_id,
  decile DESC,
  rank DESC;
""")

# COMMAND ----------

list_top_dish_type_l3 = spark.sql(
f"""
SELECT dish_type_l3
FROM {CATALOG_NAME}.{TARGET_DB_NM}.food_category_area_i_d
WHERE   num_retro_days = 7 AND 
        busi_dt = (SELECT MAX(busi_dt) FROM {CATALOG_NAME}.{TARGET_DB_NM}.food_category_area_i_d) AND
        dish_type_l3 NOT IN ('rice_noodle', 'soup', 'utensil', 'rice', 'hotpot', 'other', 'seasoning', 'chili_paste', '')
GROUP BY dish_type_l3
ORDER BY SUM(total_orders) DESC
"""
).collect()

list_top_dish_type_l3 = [row['dish_type_l3'] for row in list_top_dish_type_l3]
global list_top_dish_type_l3

# COMMAND ----------

list_top_dish_type_l3

# COMMAND ----------

dict_enc = spark.table('playground_prod.ml_dish_tagging.new_category_id_mapper') \
    .toPandas() \
    .set_index('category_name') \
    .to_dict()['category_id']

global dict_enc

# COMMAND ----------

def update_dish_types(dish_types, dish_count):

    # Global variable are max_dishes, list_top_dish_type_l3, and dict_enc
    max_dishes = 10
    
    # Fill null category in list
    if dish_count < max_dishes:
        for dish in list_top_dish_type_l3:
            if dish not in dish_types and len(dish_types) < max_dishes:
                dish_types.append(dish)
    
    dish_types = dish_types[:10]

    transformed_response = []
    for dish in dish_types:
        value = {}
        value['id'] = dict_enc[dish]
        value['name'] = dish
        transformed_response.append(value)
    
    return transformed_response

output_schema = ArrayType(StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("name", StringType(), nullable=False)
]))

update_dish_types_udf = F.udf(update_dish_types, output_schema)

# COMMAND ----------

row_top_dish = spark.createDataFrame([Row(user_id="default", 
                                         sv_dish_types=list_top_dish_type_l3[:10], 
                                         sv_dish_count=10
                                         )
                                         ])

# COMMAND ----------

row_top_dish.display()

# COMMAND ----------

df_updated = df.groupBy('user_id')\
  .agg(
      F.collect_list('dish_type_l3').alias('sv_dish_types'),
      F.count('dish_type_l3').alias('sv_dish_count')
  )\
  .unionAll(row_top_dish)\
  .withColumns({
    'gd_dish_types': F.to_json(update_dish_types_udf(F.col('sv_dish_types'), F.col('sv_dish_count'))),
    'busi_dt': F.to_date(F.lit(busi_dt))
    })

# COMMAND ----------

def convert_delta_to_feature(fe, delta_df, feature_table, primary_keys, description, mode='merge'):
    try:
        print("Check feature store is already existed or not")
        fs = FeatureStoreClient()
        fs.get_table(feature_table)
    except:
        print("Create new feature store table")
        fe.create_table(
        name=feature_table,
        primary_keys=primary_keys,
        df=delta_df,
        schema=delta_df.schema,
        description=description
    )
        
    print("Write data to feature store table")
    fe.write_table(
        name=feature_table,
        df = delta_df,
        mode = mode
    )

# COMMAND ----------

convert_delta_to_feature(fe = FeatureEngineeringClient(), 
                         delta_df = df_updated,
                         feature_table = FEATURE_TABLE, 
                         primary_keys = 'user_id', 
                         description = f"For dish bubble API call from Databricks in {CATALOG_NAME}"
                         )

# COMMAND ----------

publish_to_dynamodb(fe = FeatureEngineeringClient(), 
                    catalog = CATALOG_NAME, 
                    dynamodb_table = FEATURE_TABLE, 
                    mode='merge')