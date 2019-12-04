import concurrent.futures
import time
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, List, Union

import requests
from requests.adapters import HTTPAdapter


class Status(Enum):
    PROCESSING = 1
    SUCCESS = 2
    FAILED = 3


class UploadJob:
    def __init__(self, upload_id: str, status: Status) -> None:
        self.id = upload_id
        self.status = status

    def to_dict(self):
        return {'uploadId': self.id, 'status': self.status.name}

    @classmethod
    def from_dict(cls, data):
        renamed_data = {
            'upload_id': data['uploadId'],
            'status': Status[data['status']],
        }
        return cls(**renamed_data)


class DatasetClient:
    MAX_RETRIES = 3
    FILE_EXTENSIONS = {
        '.csv': 'text/csv',
        '.zip': 'application/zip',
        '.npy': 'application/npy',
    }

    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._session = requests.Session()
        self._session.mount('', HTTPAdapter(max_retries=self.MAX_RETRIES))
        self._session.headers.update(
            {
                'Authorization': 'Bearer %s' % token,
                'User-Agent': 'sidekick',
            }
        )

    def create_wrapper(self, name: str, description: str) -> str:
        response = self._session.post(
            url=self._url,
            headers={'Content-Type': 'application/json'},
            json={"name": name, "description": description}
        )
        response.raise_for_status()
        return response.json()['datasetWrapperId']

    def upload_file(self, filepath: Union[str, Path], wrapper_id: str) -> str:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError('File not found %s' % path)

        if path.suffix not in self.FILE_EXTENSIONS:
            supported_extensions = set(self.FILE_EXTENSIONS.keys())
            raise ValueError(
                'Valid extensions: %s. Given: %s' % (
                    supported_extensions, path.suffix)
            )
        content_type = self.FILE_EXTENSIONS[path.suffix]

        with open(path, 'rb') as file:
            data = file.read()

        print('Uploading file %s...' % path)
        response = self._session.post(
            url=self._url + '%s/upload' % wrapper_id,
            headers={'Content-Type': content_type},
            data=data
        )
        response.raise_for_status()
        print('File %s uploaded' % path)
        return response.json()['uploadId']

    def get_status(self, wrapper_id: str) -> List[UploadJob]:
        response = self._session.get(
            url=self._url + '%s/uploads' % wrapper_id,
        )
        response.raise_for_status()
        jobs = response.json()['uploadStatuses']
        return [UploadJob.from_dict(job) for job in jobs]

    def complete_upload(self, wrapper_id: str) -> None:
        response = self._session.post(
            url=self._url + '%s/upload_complete' % wrapper_id,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()


class Polling:
    def __init__(self, polling_fn: Callable[[], List[UploadJob]]) -> None:
        self.polling_fn = polling_fn
        self.successful_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        self.ongoing = True

    def update(self):
        self.ongoing = False
        jobs = self.polling_fn()

        for job in jobs:
            if job.status is Status.FAILED:
                self.failed_jobs.append(job.id)
            elif job.status is Status.SUCCESS:
                if job.id not in self.successful_jobs:
                    self.successful_jobs.append(job.id)
                    print('Job %s successfully saved.' % job.id)
            elif job.status is Status.PROCESSING:
                self.ongoing = True
            else:
                raise ValueError('Invalid state: %s' % job.status)


def create_dataset_and_upload_many_files(
    file_paths: List[str],
    url: str,
    token: str,
    dataset_name: str = 'Sidekick upload',
    dataset_description: str = 'Sidekick upload',
) -> None:

    client = DatasetClient(url, token)
    wrapper_id = client.create_wrapper(dataset_name, dataset_description)

    workers = max(10, len(file_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for file_path in file_paths:
            future = pool.submit(client.upload_file, file_path, wrapper_id)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()

    polling = Polling(partial(client.get_status, wrapper_id))
    while polling.ongoing:
        polling.update()
        if polling.failed_jobs:
            raise IOError('Failed jobs: %s' % polling.failed_jobs)
        time.sleep(1)

    client.complete_upload(wrapper_id)
