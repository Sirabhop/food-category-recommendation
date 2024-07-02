import logging
import boto3
import os
from mlflow.pyfunc import PythonModel
from boto3.dynamodb.conditions import Key, Attr

class getPersonalizedDishes(PythonModel):

    def __init__(self, keys):
        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.aws_access_key = keys[0]
        self.aws_secret_key = keys[1]

    def load_context(self, context):
        self.client = boto3.client('dynamodb', 
                            region_name='ap-southeast-1',
                            aws_access_key_id=self.aws_access_key,
                            aws_secret_access_key=self.aws_secret_key
                            )

    def __query_from_dynamo(self, user_id):
        feature_table = 'dish_types_recommendation'
        response = self.client.get_item(
            TableName=feature_table,
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
        format_res = [d['S'] for d in items]
        return  format_res

    def __postprocess_result(self, user_id, response):
        if "Item" in response:
            res = response['Item']['gd_dish_types']['L'][:10]
            format_output = self.__format_output(res)
            return {'user_id': user_id, 'recommended_items': format_output}
        else:
            query_items = self.__query_from_dynamo('default')
            res = query_items['Item']['gd_dish_types']['L'][:10]
            format_output = self.__format_output(res)
            return {'user_id': user_id, 'recommended_items': format_output}

    def predict(self, context, model_input):
        user_id = model_input['user_id'].values[0]
        query_items = self.__query_from_dynamo(user_id)
        result = self.__postprocess_result(user_id, query_items)
        return result