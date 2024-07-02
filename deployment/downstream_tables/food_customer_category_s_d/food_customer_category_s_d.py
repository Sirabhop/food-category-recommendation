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
dbutils.widgets.text("target_tbl_nm", "food_customer_category_s_d")
dbutils.widgets.text("catalog", "prod")
dbutils.widgets.text("busi_dt", busi_dt)

layer = dbutils.widgets.get('layer')
target_db_nm = dbutils.widgets.get("target_db_nm")
src_sys = dbutils.widgets.get('src_sys')
target_tbl_nm = dbutils.widgets.get("target_tbl_nm")
catalog = dbutils.widgets.get("catalog")
busi_dt = dbutils.widgets.get("busi_dt")

# COMMAND ----------

df_base = spark.sql(
f"""
SELECT
    a.user_id,
    c.dish_type_l3,
    CAST(c.category_id AS INT) AS category_id,
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
    c.category_id;
"""
)

df_base.createOrReplaceTempView("based_preference_table")

# COMMAND ----------

df_customer_preference_raw = spark.sql(
"""
WITH DistinctOrders (
    SELECT 
        DISTINCT user_id,
        count_order
    FROM 
        based_preference_table
),
RankedOrders AS (
    SELECT 
        user_id,
        count_order,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY count_order ASC) AS rank,
        COUNT(*) OVER (PARTITION BY user_id) AS total_distinct_orders
    FROM
        DistinctOrders
)
SELECT
    b.user_id,
    b.count_order,
    b.recent_order_dt,
    b.dish_type_l3,
    b.category_id,
    CEILING(a.rank * 10.0 / a.total_distinct_orders) AS decile,
    ROW_NUMBER() OVER (
        PARTITION BY b.user_id, CEILING(a.rank * 10.0 / a.total_distinct_orders) 
        ORDER BY b.recent_order_dt ASC) AS rank 
FROM
    RankedOrders AS a
RIGHT JOIN
    based_preference_table AS b
    ON a.user_id = b.user_id AND a.count_order = b.count_order;
"""
)

# COMMAND ----------

df_customer_preference_raw = df_customer_preference_raw \
    .replace("", None) \
    .dropna(subset=["dish_type_l3"], how="all") \
    .withColumn("percentile", F.concat_ws("", F.lit("P"), F.col("decile"))) \
    .withColumn("busi_dt", F.to_date(F.lit(busi_dt)))

# COMMAND ----------

df_customer_preference_raw.write.mode('overWrite').option("mergeSchema", "true").saveAsTable('prod.gd_food.food_customer_category_s_d')