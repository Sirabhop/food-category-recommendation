# Databricks notebook source
from datetime import timedelta, datetime

import pyspark.sql.functions as F
import pyspark.sql.types as T

busi_dt = str(datetime.today().date() - timedelta(1))
busi_dt

# COMMAND ----------

# Set up variable
dbutils.widgets.text('layer', 'gd')
dbutils.widgets.text('src_sys', 'food')
dbutils.widgets.text("target_db_nm", "gd_food")
dbutils.widgets.text("target_tbl_nm", "food_category_area_i_d")
dbutils.widgets.text("catalog", "prod")
dbutils.widgets.text("busi_dt", busi_dt)

layer = dbutils.widgets.get('layer')
target_db_nm = dbutils.widgets.get("target_db_nm")
src_sys = dbutils.widgets.get('src_sys')
target_tbl_nm = dbutils.widgets.get("target_tbl_nm")
catalog = dbutils.widgets.get("catalog")
busi_dt = dbutils.widgets.get("busi_dt")

# COMMAND ----------

num_retro_days_list = [1, 7, 30, 120, 365]

# COMMAND ----------

def get_data(run_date, interval):
    print('cal on', run_date, 'with interval', interval, 'days')
    df = spark.sql(
f"""
WITH baseOrder AS (
    SELECT  a.order_id,
            a.total_amount,
            c.address_subdistrict,
            e.dish_type_l3,
            e.category_id,
            ROW_NUMBER() OVER (PARTITION BY a.order_id ORDER BY a.updt_dttm DESC) AS row_num
    FROM prod.sv_aurora_major_ordering_vw.order_transaction_i_d a
    LEFT JOIN prod.sv_aurora_major_merchant_vw.shop_a_d b
      ON a.shop_id = b.shop_id
    LEFT JOIN prod.sv_aurora_major_merchant_vw.address_a_d c
      ON c.address_id = b.address_id
    RIGHT JOIN 
        prod.sv_aurora_major_ordering_vw.order_item_a_d AS d
    ON 
        a.order_id = d.order_id
    LEFT JOIN ( 
        SELECT 
            menu_id,
            new_dish_type_l3 AS dish_type_l3,
            new_category_id AS category_id
        FROM playground_prod.ml_dish_tagging.dish_tagging
    ) AS e
    ON 
        d.menu_id = e.menu_id
    WHERE   DATE(a.order_date) >= DATE_SUB(DATE("{run_date}"), {interval}) AND
            DATE(a.order_date) <= DATE("{run_date}") AND
            e.dish_type_l3 IS NOT NULL AND
            e.dish_type_l3 != ''
    ORDER BY a.order_id
)
SELECT  address_subdistrict AS subdistrict,
        dish_type_l3,
        CAST(category_id AS INT),
        CAST(sum(total_amount) AS DOUBLE) AS total_amount,
        CAST(count(order_id) AS LONG) AS total_orders,
        DATE("{run_date}") AS run_date,
        {interval} AS num_retro_days,
        DATE("{run_date}") AS busi_dt
FROM    baseOrder
WHERE   row_num = 1 AND
        dish_type_l3 IS NOT NULL
GROUP BY address_subdistrict, dish_type_l3, category_id
    """
    )

    return df

# COMMAND ----------

today_date_list = []
for i in range(1):
    today_date = datetime.now().date() - timedelta(i+1)
    today_date_list.append(today_date)
    
today_date_list.reverse()

print(today_date_list[0], today_date_list[-1])

# COMMAND ----------

final_df = None 

for i, today_date in enumerate(today_date_list):
    for num_retro_days in num_retro_days_list:

        df_to_upload = get_data(today_date, num_retro_days)
        df_to_upload.cache()
        
        if final_df is None:
            final_df = df_to_upload
        else:
            final_df = final_df.unionByName(df_to_upload)

    # if i == 0:
    #     final_df.write.mode('overWrite').saveAsTable('prod.gd_food.food_category_area_i_d')

    # else:
    final_df.write.mode('append').option("mergeSchema", "true").saveAsTable('prod.gd_food.food_category_area_i_d')
