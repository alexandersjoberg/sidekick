from zipfile import ZipFile

import pytest
import responses
from PIL import Image

from sidekick.dataset_client import Dataset, Status, UploadJob


@pytest.fixture
def client():
    return Dataset(url='http://localhost', token='')


@pytest.fixture
def csv_file(tmp_path):
    filepath = tmp_path / 'mock.csv'
    with filepath.open('w') as file:
        file.write('mock')
    return filepath


@pytest.fixture
def jpeg_file(tmp_path):
    filepath = tmp_path / 'mock.jpeg'
    Image.new(mode='RGB', size=(10, 10)).save(filepath)
    return filepath


@pytest.fixture
def zip_file(tmp_path):
    filepath = tmp_path / 'mock.zip'
    with ZipFile(filepath, 'w') as zipfile:
        zipfile.writestr('mock.txt', 'mock')
    return filepath


class TestDeserializeJobUpload:
    upload_id = '9b6c9f7c9bc74c13a462f74b7dfb497b'

    def test_success_job(self):
        status = 'SUCCESS'
        job_dict = {'uploadId': self.upload_id, 'status': status}
        job_object = UploadJob.from_dict(job_dict)
        assert job_object.status is Status[status]
        assert job_object.id == self.upload_id
        assert job_object.message is None

    def test_failed_job(self):
        status = 'FAILED'
        message = 'mock'
        job_dict = {
            'uploadId': self.upload_id, 'status': status, 'message': message
        }
        job_object = UploadJob.from_dict(job_dict)
        assert job_object.status is Status[status]
        assert job_object.id == self.upload_id
        assert job_object.message == message


class TestDatasetClient:
    def test_url_with_trailing_slash(self):
        url = 'http://localhost'
        client1 = Dataset(url=url, token='')
        client2 = Dataset(url=url + '/', token='')
        assert client1.url == url
        assert client2.url == url

    def test_non_existent_files(self, client):
        with pytest.raises(FileNotFoundError):
            client.upload_files(['mock.csv'])

    def test_files_with_invalid_extension(self, client, jpeg_file):
        with pytest.raises(ValueError):
            client.upload_files([jpeg_file])

    @responses.activate
    def test_upload_files(self, client, csv_file, zip_file):
        wrapper_id = "wrapper_id"
        upload_id1 = 'id1'
        upload_id2 = 'id2'

        responses.add(
            method=responses.POST,
            url=client.url,
            json={"datasetWrapperId": wrapper_id}
        )

        responses.add(
            method=responses.POST,
            url='%s/%s/upload' % (client.url, wrapper_id),
            json={'uploadId': upload_id1},
        )

        responses.add(
            method=responses.POST,
            url='%s/%s/upload' % (client.url, wrapper_id),
            json={'uploadId': upload_id2},
        )

        responses.add(
            method=responses.GET,
            url='%s/%s/uploads' % (client.url, wrapper_id),
            json={
                'uploadStatuses': [
                    {
                        'uploadId': upload_id1,
                        'status': 'PROCESSING',
                    },
                    {
                        'uploadId': upload_id2,
                        'status': 'SUCCESS',
                    }
                ]
            },
        )

        responses.add(
            method=responses.GET,
            url='%s/%s/uploads' % (client.url, wrapper_id),
            json={
                'uploadStatuses': [
                    {
                        'uploadId': upload_id1,
                        'status': 'SUCCESS',
                    },
                    {
                        'uploadId': upload_id2,
                        'status': 'SUCCESS',
                    }
                ]
            },
        )

        responses.add(
            method=responses.POST,
            url='%s/%s/upload_complete' % (client.url, wrapper_id),
            status=204,
        )

        client.upload_files([csv_file, zip_file], num_threads=1)
        assert len(responses.calls) == 6

    @responses.activate
    def test_failed_upload_job(self, client, csv_file, zip_file):
        wrapper_id = 'wrapper_id'
        upload_id1 = 'id1'
        upload_id2 = 'id2'

        responses.add(
            method=responses.POST,
            url=client.url,
            json={'datasetWrapperId': wrapper_id}
        )

        responses.add(
            method=responses.POST,
            url='%s/%s/upload' % (client.url, wrapper_id),
            json={'uploadId': upload_id1},
        )

        responses.add(
            method=responses.POST,
            url='%s/%s/upload' % (client.url, wrapper_id),
            json={'uploadId': upload_id2},
        )

        responses.add(
            method=responses.GET,
            url='%s/%s/uploads' % (client.url, wrapper_id),
            json={
                'uploadStatuses': [
                    {
                        'uploadId': upload_id1,
                        'status': 'PROCESSING',
                    },
                    {
                        'uploadId': upload_id2,
                        'status': 'FAILED',
                        'message': 'mock'
                    }
                ]
            },
        )

        with pytest.raises(IOError):
            client.upload_files([csv_file, zip_file], num_threads=1)
