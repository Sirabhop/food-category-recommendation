# Databricks notebook source
import pyspark.sql.functions as F
from datetime import datetime, timedelta

# COMMAND ----------

busi_dt = str(datetime.today().date() - timedelta(1))
busi_dt

# COMMAND ----------

# Set up variable
dbutils.widgets.text('layer', 'gd')
dbutils.widgets.text('src_sys', 'food')
dbutils.widgets.text("target_db_nm", "gd_food")
dbutils.widgets.text("target_tbl_nm", "food_shop_category_s_d")
dbutils.widgets.text("catalog", "prod")
dbutils.widgets.text("busi_dt", busi_dt)

LAYER_NAME = dbutils.widgets.get('layer')
CATALOG_NAME = dbutils.widgets.get("catalog")
SRC_SYS = dbutils.widgets.get('src_sys')
BUSI_DT = dbutils.widgets.get("busi_dt")
TARGET_DB_NM = dbutils.widgets.get("target_db_nm")
TARGET_TBL_NM = dbutils.widgets.get("target_tbl_nm")

# COMMAND ----------

df_shop = spark.sql(
f"""    
WITH OrderTransactions AS (
    SELECT  shop_id, 
            shop_name, 
            order_id,
            ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY updt_dttm DESC) AS row_num
    FROM prod.sv_aurora_major_ordering_vw.order_transaction_i_d
),
OrderCounts AS (
    SELECT
        a.shop_id, 
        a.shop_name,
        c.dish_type_l3,
        c.category_id,
        COUNT(a.order_id) AS order_count
    FROM OrderTransactions AS a
    LEFT JOIN 
        prod.sv_aurora_major_ordering_vw.order_item_a_d AS b
    ON 
        a.order_id = b.order_id
    LEFT JOIN ( 
        SELECT 
            menu_id,
            new_dish_type_l3 AS dish_type_l3,
            new_category_id AS category_id
        FROM playground_prod.ml_dish_tagging.dish_tagging
    ) AS c
    ON 
        b.menu_id = c.menu_id
    WHERE   a.row_num = 1 AND
            c.dish_type_l3 IS NOT NULL
    GROUP BY 
        a.shop_id, a.shop_name, c.dish_type_l3, c.category_id
), RankedDishes AS (
    SELECT
        shop_id,
        shop_name,
        dish_type_l3,
        category_id,
        order_count,
        ROW_NUMBER() OVER(PARTITION BY shop_id ORDER BY order_count DESC) as rank
    FROM
        OrderCounts
)

SELECT 
    shop_id,
    shop_name,
    dish_type_l3 AS shop_category,
    CAST(category_id AS INT) AS category_id,
    order_count,
    DATE('{busi_dt}') AS busi_dt
FROM
    RankedDishes
WHERE
    rank = 1;
""")

# COMMAND ----------

df_prev = spark.table('prod.gd_food.food_shop_category_s_d')

df_changed = df_shop.join(df_prev, on=['shop_id', 'shop_name', 'shop_category', 'category_id', 'order_count'], how='left_anti')
df_changed = df_changed.withColumn('busi_dt', (F.current_date() - F.lit(1)).cast("date"))

# COMMAND ----------

df_changed.write.mode('overWrite').saveAsTable('playground_prod.ml_dish_tagging.food_shop_category_h_d')
df_shop.write.mode('overWrite').option("mergeSchema", "true").saveAsTable('prod.gd_food.food_shop_category_s_d')