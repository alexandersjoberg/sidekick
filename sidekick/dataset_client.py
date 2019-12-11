import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter


class Status(Enum):
    PROCESSING = 1
    SUCCESS = 2
    FAILED = 3


class UploadJob:
    def __init__(
        self, upload_id: str, status: Status, message: str
    ) -> None:
        self.id = upload_id
        self.status = status
        self.message = message

    @classmethod
    def from_dict(cls, data):
        return cls(
            upload_id=data['uploadId'],
            status=Status[data['status']],
            message=data.get('message'),
        )


class Dataset:
    """Client for the Dataset API."""

    _MAX_RETRIES = 3
    _EXTENSION_MAPPING = {
        '.csv': 'text/csv',
        '.zip': 'application/zip',
        '.npy': 'application/npy',
    }
    VALID_EXTENSIONS = set(_EXTENSION_MAPPING.keys())

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip('/')
        self._session = requests.Session()
        self._session.mount('', HTTPAdapter(max_retries=self._MAX_RETRIES))
        self._session.headers.update(
            {
                'Authorization': 'Bearer %s' % token,
                'User-Agent': 'sidekick',
            }
        )

    def upload_files(
        self,
        filepaths: List[str],
        name: str = 'Sidekick upload',
        description: str = 'Sidekick upload',
    ) -> None:
        """Creates a dataset and uploads files to it.

        Args:
            filepaths: List of files to upload to the dataset.
            name: Name of the dataset.
            description: Description of the dataset.

        Raises:
            FileNotFoundError: One or more filepaths not found.
            ValueError: One or more files have a non supported extension.
            IOError: Error occurred while saving files in dataset.

        """

        paths = [Path(path).resolve() for path in filepaths]
        self._validate_paths(paths)
        wrapper_id = self._create_wrapper(name, description)
        jobs_mapping = self._stage_files(paths, wrapper_id)
        self._wait_until_completed(wrapper_id, jobs_mapping)
        self._complete_upload(wrapper_id)

    def _create_wrapper(self, name: str, description: str) -> str:
        response = self._session.post(
            url=self.url,
            headers={'Content-Type': 'application/json'},
            json={'name': name, 'description': description}
        )
        response.raise_for_status()
        return response.json()['datasetWrapperId']

    def _get_status(self, wrapper_id: str) -> List[UploadJob]:
        response = self._session.get(
            url=self.url + '/%s/uploads' % wrapper_id,
        )
        response.raise_for_status()
        jobs = response.json()['uploadStatuses']
        return [UploadJob.from_dict(job) for job in jobs]

    def _complete_upload(self, wrapper_id: str) -> None:
        response = self._session.post(
            url=self.url + '/%s/upload_complete' % wrapper_id,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()

    def _stage_files(
        self, filepaths: List[Path], wrapper_id: str
    ) -> Dict[str, Path]:

        workers = min(10, len(filepaths))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for path in filepaths:
                future = pool.submit(self._stage_file, path, wrapper_id)
                futures.append(future)
            job_ids = [future.result() for future in futures]
        return dict(zip(job_ids, filepaths))

    def _stage_file(self, filepath: Path, wrapper_id: str) -> str:
        content_type = self._EXTENSION_MAPPING[filepath.suffix]

        with filepath.open('rb') as file:
            data = file.read()

        print('Staging file: %s...' % filepath)
        response = self._session.post(
            url=self.url + '/%s/upload' % wrapper_id,
            headers={'Content-Type': content_type},
            data=data
        )
        response.raise_for_status()
        print('File staged: %s' % filepath)
        return response.json()['uploadId']

    def _validate_paths(self, paths: List[Path]) -> None:
        """Validates that paths exist and have a supported extension."""

        not_found = [str(path) for path in paths if not path.exists()]
        if not_found:
            raise FileNotFoundError('Files not found: %s' % set(not_found))

        invalid_extension = [
            str(path) for path in paths
            if path.suffix not in self.VALID_EXTENSIONS
        ]
        if invalid_extension:
            raise ValueError(
                'Valid extensions: %s. Given: %s' % (
                    self.VALID_EXTENSIONS, set(invalid_extension))
            )

    def _wait_until_completed(
        self, wrapper_id: str, job_mapping: Dict[str, Path]
    ) -> None:
        """Waits until all jobs are saved."""

        ongoing = True
        successful_jobs = []  # type: List[str]
        latest_reporting_time = datetime.now()

        while ongoing:
            ongoing = False
            jobs = self._get_status(wrapper_id)
            for job in jobs:
                if job.status is Status.FAILED:
                    raise IOError(
                        'Error saving file: %s, message: %s' % (
                            job_mapping[job.id], job.message
                        )
                    )
                elif job.status is Status.SUCCESS:
                    if job.id not in successful_jobs:
                        successful_jobs.append(job.id)
                        filename = job_mapping[job.id]
                        print('File uploaded: %s' % filename)
                else:  # status is PROCESSING:
                    ongoing = True

            if ongoing:
                time.sleep(1)

            elapsed_time = datetime.now() - latest_reporting_time
            if elapsed_time.seconds > 60 * 5:
                print('Uploading files....')
                latest_reporting_time = datetime.now()

        print('All files were uploaded.')
