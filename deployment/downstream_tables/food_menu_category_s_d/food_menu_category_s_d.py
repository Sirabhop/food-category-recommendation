# Databricks notebook source
import pyspark.sql.functions as F
from datetime import datetime, timedelta

busi_dt = str(datetime.today().date() - timedelta(1))
busi_dt

# COMMAND ----------

# Set up variable
dbutils.widgets.text('layer', 'gd')
dbutils.widgets.text('src_sys', 'food')
dbutils.widgets.text("target_db_nm", "gd_food")
dbutils.widgets.text("target_tbl_nm", "food_menu_category_s_d")
dbutils.widgets.text("catalog", "prod")
dbutils.widgets.text("busi_dt", busi_dt)

LAYER_NAME = dbutils.widgets.get('layer')
CATALOG_NAME = dbutils.widgets.get("catalog")
SRC_SYS = dbutils.widgets.get('src_sys')
BUSI_DT = dbutils.widgets.get("busi_dt")
TARGET_DB_NM = dbutils.widgets.get("target_db_nm")
TARGET_TBL_NM = dbutils.widgets.get("target_tbl_nm")

# COMMAND ----------

df_base = spark.sql(
f"""
SELECT
    a.user_id,
    c.dish_type_l3,
    c.category_id,
    COUNT(*) AS count_order,
    MAX(a.order_date) AS recent_order_dt
FROM
    prod.sv_aurora_major_ordering_vw.order_transaction_i_d AS a
LEFT JOIN prod.sv_aurora_major_ordering_vw.order_item_a_d AS b ON a.order_id = b.order_id
LEFT JOIN (
    SELECT
        menu_id,
        new_dish_type_l3 AS dish_type_l3,
        new_category_id AS category_id
    FROM
        playground_prod.ml_dish_tagging.dish_tagging
) AS c ON b.menu_id = c.menu_id
WHERE   a.order_status = 'COMPLETE' AND
        c.dish_type_l3 IS NOT NULL
GROUP BY
    a.user_id,
    c.dish_type_l3,
    c.category_id
"""
)

df_base.createOrReplaceTempView("based_preference_table")

# COMMAND ----------

df_menu_preference_raw = spark.sql(
"""
WITH DistinctOrders AS (
    SELECT 
        user_id,
        dish_type_l3,
        category_id,
        COUNT(*) AS count_order
    FROM 
        based_preference_table
    GROUP BY
        user_id,
        dish_type_l3,
        category_id
),
RankedOrders AS (
    SELECT 
        user_id,
        dish_type_l3,
        category_id,
        count_order,
        ROW_NUMBER() OVER (PARTITION BY dish_type_l3 ORDER BY count_order ASC) AS rank,
        COUNT(*) OVER (PARTITION BY dish_type_l3) AS total_distinct_orders
    FROM
        DistinctOrders
)
SELECT
    b.user_id,
    b.count_order,
    b.dish_type_l3,
    CAST(b.category_id AS INT) AS category_id,
    CEILING(a.rank * 10.0 / a.total_distinct_orders) AS decile
FROM
    RankedOrders AS a
RIGHT JOIN
    based_preference_table AS b
    ON a.user_id = b.user_id AND a.dish_type_l3 = b.dish_type_l3;
""")

# COMMAND ----------

df_menu_preference_raw = df_menu_preference_raw \
    .replace("", None) \
    .dropna(subset=["dish_type_l3"], how="all") \
    .withColumn("percentile", F.concat_ws("", F.lit("P"), F.col("decile"))) \
    .withColumn("busi_dt", F.to_date(F.lit(busi_dt)))

# COMMAND ----------

df_menu_preference_raw.write.mode('overWrite').option("mergeSchema", "true").saveAsTable(f'{CATALOG_NAME}.{TARGET_DB_NM}.{TARGET_TBL_NM}')