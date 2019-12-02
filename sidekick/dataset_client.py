import requests
from requests.adapters import HTTPAdapter
import json
from typing import List
import os

from enum import Enum


class UploadStatus(Enum):
    PENDING = 1
    PROCESSING = 2
    SUCCESS = 3
    FAILED = 4


class DatasetClient:
    MAX_RETRIES = 3

    def __init__(self, url: str, token: str, verbose=False) -> None:
        self._headers = {'Authorization': 'Bearer ' + token}
        self._url = url
        self._verbose = verbose
        self._session = requests.Session()
        self._session.mount('', HTTPAdapter(max_retries=self.MAX_RETRIES))
        self._session.headers.update({'User-Agent': 'sidekick'})

    def _call_avoin(self, resource: str, verb: str, payload=None) -> str:
        response = None
        if self._verbose: print("Calling " + self._url + resource)
        if verb == 'post':
            response = self._session.post(
                url=self._url + resource,
                headers=self._headers,
                data=payload
            )
        elif verb == 'get':
            response = self._session.get(
                url=self._url + resource,
                headers=self._headers
            )
        else:
            raise ValueError('HTTP verb ' + verb + ' not supported.')

        print("Response status: " + str(response.status_code) + ", " + response.text) if self._verbose else 0
        return response.text

    def _create_wrapper(self, name='Sidekick upload', description='Sidekick upload'):
        self._headers['Content-Type'] = 'application/json'
        return self._call_avoin('', 'post',
                                "{\"name\": \"" + name + "\",\"description\": \"" + description + "\"}")

    def _upload_file(self, filepath: str, wrapper_id: str):
        fileobj = open(filepath, 'rb').read()
        self._headers['Content-Type'] = self._get_content_type(filepath)
        return self._call_avoin(f'/{wrapper_id}/upload', 'post', fileobj)

    def _get_upload_status(self, wrapper_id: str):
        self._headers['Content-Type'] = None
        return self._call_avoin(f'/{wrapper_id}/uploads', 'get')

    def _complete_upload(self, wrapper_id: str):
        self._headers['Accept'] = "application/json"
        return self._call_avoin(f'/{wrapper_id}/upload_complete', 'post')

    def create_upload_many(self, files: List[str], dataset_name='Sidekick upload',
                           dataset_description='Sidekick upload'):
        response = self._create_wrapper(dataset_name, dataset_description)
        wrapper_id = json.loads(response)['datasetWrapperId']
        for file in files:
            self._upload_file(file, wrapper_id)

        upload_ongoing = True
        while upload_ongoing:
            response = self._get_upload_status(wrapper_id)
            if self._has_failed_uploads(response):
                raise ValueError('One or more uploads failed. Response: ' + response)
            upload_ongoing = self._has_ongoing_uploads(response)

        return self._complete_upload(wrapper_id)

    def _has_ongoing_uploads(self, response: str):
        statuses = self._get_statuses(response)
        return UploadStatus.PROCESSING in statuses or UploadStatus.PENDING in statuses

    def _has_failed_uploads(self, response: str):
        statuses = self._get_statuses(response)
        return UploadStatus.FAILED in statuses

    def _get_statuses(self, response: str) -> List[UploadStatus]:
        uploads = json.loads(response)['uploadStatuses']
        upload_status = []
        for upload in uploads:
            upload_status.append(UploadStatus[upload['status']])

        return upload_status

    def _get_content_type(self, filepath: str):
        file_extension = os.path.splitext(filepath)[1]
        content_type = ""
        if file_extension == '.csv':
            content_type = 'text/csv'
        elif file_extension == '.zip':
            content_type = 'application/zip'
        elif file_extension == '.npy':
            content_type = 'application/npy'
        else:
            raise ValueError('Not able to set content-type from file name')

        print("Setting Content-Type to " + content_type) if self._verbose else 0
        return content_type