import json
import requests

class databricks_api():

    def __init__(self, env_url, token):
        self.env_url = env_url
        self.headers = {"Authorization": f"Bearer {token}"} 

    def _get_job_info(self, job_id):
        endpoint = f'{self.env_url}/api/2.1/jobs/get'
        payload = {'job_id': int(job_id)}
        try:
            response = requests.get(endpoint, headers=self.headers, params=payload)
            response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX or 5XX
            return response.json(), 'pass'
        except requests.exceptions.HTTPError as err:
            return {'error': f'HTTP error occurred: {err}'}, 'error'
        except requests.exceptions.RequestException as e:
            return {'error': f'Request exception occurred: {e}'}, 'error'
        
    def _update_job_tag(self, job_id, new_tag):
        url = f"{self.env_url}/api/2.1/jobs/update"
        payload = {
            "job_id": int(job_id),
            "new_settings": {
                "tags": {
                    "cicd-tag": new_tag
                }
            }
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX or 5XX
            return response.json(), 'pass'
        except requests.exceptions.HTTPError as err:
            return {'error': f'HTTP error occurred: {err}'}, 'error'
        except requests.exceptions.RequestException as e:
            return {'error': f'Request exception occurred: {e}'}, 'error'
        
    def _create_job(self, cluster_id):
        url = f"{self.env_url}/api/2.1/jobs/create"
        with open("databricks_create_job_template.json", "r") as f:
            config_dict = json.load(f)
            config_dict['tasks'][0]['existing_cluster_id'] = cluster_id
        payload = json.dumps(config_dict)

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX or 5XX
            return response.json(), 'pass'
        except requests.exceptions.HTTPError as err:
            return {'error': f'HTTP error occurred: {err}'}, 'error'
        except requests.exceptions.RequestException as e:
            return {'error': f'Request exception occurred: {e}'}, 'error'
        
    def check_and_tag(self, job_id, cluster_id, gitlab_running_code):

        # API get job information
        job_info, status = self._get_job_info(job_id=job_id)

        # Check if the job has been set
        if status != 'pass':
            status = self._create_job(cluster_id=cluster_id)
        elif status == 'pass':
            status = self._update_job_tag(job_id=job_id, new_tag=gitlab_running_code)
        
        


