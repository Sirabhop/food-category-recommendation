from databricks import feature_store

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd

from databricks.sdk.runtime import dbutils
RANDOM_SEED = dbutils.widgets.get('RANDOM_SEED')

def clean_sentence(name):
    CHAR = [
        u'A', u'B', u'C', u'D', u'E', ' ',
        u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
        u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z',
        u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
        u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
        u'z', u'ก', u'ข', u'ฃ', u'ค', u'ฅ', u'ฆ', u'ง', u'จ', u'ฉ', u'ช',
        u'ซ', u'ฌ', u'ญ', u'ฎ', u'ฏ', u'ฐ', u'ฑ', u'ฒ', u'ณ', u'ด', u'ต', u'ถ', u'ท',
        u'ธ', u'น', u'บ', u'ป', u'ผ', u'ฝ', u'พ', u'ฟ', u'ภ', u'ม', u'ย', u'ร', u'ฤ',
        u'ล', u'ว', u'ศ', u'ษ', u'ส', u'ห', u'ฬ', u'อ', u'ฮ', u'ฯ', u'ะ', u'ั', u'า',
        u'ำ', u'ิ', u'ี', u'ึ', u'ื', u'ุ', u'ู', u'ฺ', u'เ', u'แ', u'โ', u'ใ', u'ไ',
        u'ๅ', u'ๆ', u'็', u'่', u'้', u'๊', u'๋', u'์', u'ํ', u'‘', u'’', u'\ufeff'
        ]
    return ''.join([s for s in name if s in CHAR])

def load_data(feature_store_client, table_name):
    """
        Input:
        table_name (string) = playground_prod.ml_dish_tagging.training_data
    """
    # Get unique 3rd layer dish type
    training_set = feature_store_client.read_table(table_name)
    training_pd = training_set.toPandas()

    # Encode dish type
    dish_type_list = training_pd.dish_type_l3.unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(dish_type_list)
    training_pd['dish_type_l3_encoded'] = label_encoder.transform(training_pd.dish_type_l3.values)

    # Filter characters
    training_pd['dish_name'] = [clean_sentence(s) for s in training_pd.dish_name.values]

    # Train test split
    X, X_test, y, y_test = train_test_split(training_pd.dish_name.values, 
                                            training_pd.dish_type_l3_encoded.values, 
                                            random_state=int(RANDOM_SEED), test_size=0.2, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, 
                                                      y, 
                                                      random_state=int(RANDOM_SEED), test_size=0.15, shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, training_set, dish_type_list, label_encoder