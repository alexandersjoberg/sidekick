import json
import os
import time
import urllib.parse
from enum import Enum
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter


class UploadState(Enum):
    PENDING = 1
    PROCESSING = 2
    SUCCESS = 3
    FAILED = 4


def _get_statuses(response: str) -> List[UploadState]:
    uploads = json.loads(response)['uploadStatuses']
    upload_status = [UploadState[upload['status']] for upload in uploads]
    if _has_failed_uploads(upload_status):
        raise ValueError(
            'One or more uploads failed. Response: ' + response
        )

    return upload_status


def _has_ongoing_uploads(states: List[UploadState]):
    return UploadState.PROCESSING in states or UploadState.PENDING in states


def _has_failed_uploads(states: List[UploadState]):
    return UploadState.FAILED in states


class DatasetClient:
    MAX_RETRIES = 3

    def __init__(self, url: str, token: str, verbose=False) -> None:
        self._token = token
        self._url = url
        self._verbose = verbose
        self._session = requests.Session()
        self._session.mount('', HTTPAdapter(max_retries=self.MAX_RETRIES))
        self._session.headers.update({'User-Agent': 'sidekick'})

    def _call_avoin(self, resource: str, verb: str,
                    headers: Dict[str, Any], payload=None) -> str:
        url = urllib.parse.urljoin(self._url, resource)
        if self._verbose:
            print("Calling " + url)
        if verb == 'post':
            response = self._session.post(
                url=url,
                headers=headers,
                data=payload
            )
        elif verb == 'get':
            response = self._session.get(
                url=url,
                headers=headers
            )

        response.raise_for_status()
        if self._verbose:
            print("Response status: " +
                  str(response.status_code) + ", " +
                  response.text)
        return response.text

    def _create_wrapper(self, name='Sidekick upload',
                        description='Sidekick upload'):
        headers = self._set_headers('application/json')
        payload = {"name": name, "description": description}
        json_payload = json.dumps(payload)
        if self._verbose:
            print("Calling " + self._url)

        response = self._session.post(
            url=self._url,
            headers=headers,
            data=json_payload
        )

        response.raise_for_status()
        if self._verbose:
            print("Response status: " +
                  str(response.status_code) + ", " +
                  response.text)

        return response.text

    def _upload_file(self, filepath: str, wrapper_id: str):
        with open(filepath, 'rb') as file:
            content = file.read()
            headers = self._set_headers(self._get_content_type(filepath))
            url = self._url + f'{wrapper_id}/upload'
            if self._verbose:
                print("Calling " + url)

            response = self._session.post(
                url=url,
                headers=headers,
                data=content
            )

            response.raise_for_status()
            if self._verbose:
                print("Response status: " +
                      str(response.status_code) + ", " +
                      response.text)

            return response.text

    def _get_upload_status(self, wrapper_id: str):
        headers = self._set_headers()
        response = self._session.get(
            url=self._url + f'{wrapper_id}/uploads',
            headers=headers
        )

        response.raise_for_status()
        if self._verbose:
            print("Response status: " +
                  str(response.status_code) + ", " +
                  response.text)

        return response.text

    def _complete_upload(self, wrapper_id: str):
        headers = self._set_headers('application/json')
        response = self._session.post(
            url=self._url + f'{wrapper_id}/upload_complete',
            headers=headers
        )

        response.raise_for_status()
        if self._verbose:
            print("Response status: " +
                  str(response.status_code) + ", " +
                  response.text)

        return response.text

    def create_upload_many(self, files: List[str],
                           dataset_name='Sidekick upload',
                           dataset_description='Sidekick upload'):
        response = self._create_wrapper(dataset_name, dataset_description)
        wrapper_id = json.loads(response)['datasetWrapperId']
        for file in files:
            self._upload_file(file, wrapper_id)

        upload_ongoing = True
        while upload_ongoing:
            time.sleep(1)
            statuses = _get_statuses(self._get_upload_status(wrapper_id))
            upload_ongoing = _has_ongoing_uploads(statuses)

        return self._complete_upload(wrapper_id)

    def _get_content_type(self, filepath: str) -> str:
        file_extension = os.path.splitext(filepath)[1]
        if file_extension == '.csv':
            content_type = 'text/csv'
        elif file_extension == '.zip':
            content_type = 'application/zip'
        elif file_extension == '.npy':
            content_type = 'application/npy'
        else:
            raise ValueError('Not able to set content-type from file name')

        if self._verbose:
            print("Setting Content-Type to " + content_type)

        return content_type

    def _set_headers(self, content_type: str = None) -> Dict[str, Any]:
        return {'Authorization': 'Bearer ' + self._token,
                'Content-Type': content_type}
