import csv
from zipfile import ZipFile

import pytest
import responses

from sidekick.dataset_client import DatasetClient, Polling, Status, UploadJob


def test_upload_job():
    upload_id = '9b6c9f7c9bc74c13a462f74b7dfb497b'
    status = 'SUCCESS'
    job_dict = {'uploadId': upload_id, 'status': status}
    job_object = UploadJob.from_dict(job_dict)
    assert job_object.status is Status[status]
    assert job_object.id == upload_id
    assert job_object.to_dict() == job_dict


class TestDatasetClient:
    url = 'http://localhost/'

    @responses.activate
    def test_create_wrapper(self):
        client = DatasetClient(url=self.url, token='')
        wrapper_id = "b46ce0d238944007bfd8b0877fb4625e"

        responses.add(
            method=responses.POST,
            url=self.url,
            json={"datasetWrapperId": wrapper_id}
        )
        assert wrapper_id == client.create_wrapper(name='', description='')

    @responses.activate
    def test_complete_upload(self):
        client = DatasetClient(url=self.url, token='')
        wrapper_id = "b46ce0d238944007bfd8b0877fb4625e"

        responses.add(
            method=responses.POST,
            url='%s%s/upload_complete' % (self.url, wrapper_id),
            status=204,
        )
        client.complete_upload(wrapper_id)

    @responses.activate
    def test_get_status(self):
        client = DatasetClient(url=self.url, token='')
        wrapper_id = "b46ce0d238944007bfd8b0877fb4625e"
        upload_id = 'c9500e7fc3fc41aeac2f34c1805d3e8e'
        status = 'SUCCESS'
        mock_response = {
            'uploadStatuses': [
                {
                    'uploadId': upload_id,
                    'status': status,
                }
            ]
        }

        responses.add(
            method=responses.GET,
            url='%s%s/uploads' % (self.url, wrapper_id),
            json=mock_response,
        )
        output = client.get_status(wrapper_id)
        assert len(output) == 1
        assert output[0].status is Status[status]
        assert output[0].id == upload_id

    @responses.activate
    def test_file_upload(self, tmp_path):
        client = DatasetClient(url=self.url, token='')
        wrapper_id = "b46ce0d238944007bfd8b0877fb4625e"
        upload_id = 'c9500e7fc3fc41aeac2f34c1805d3e8e'

        responses.add(
            method=responses.POST,
            url='%s%s/upload' % (self.url, wrapper_id),
            json={'uploadId': upload_id},
        )

        with pytest.raises(FileNotFoundError):
            client.upload_file(tmp_path / 'mock.csv', wrapper_id)

        invalid_extension_dataset = tmp_path / 'mock.fds'
        with open(invalid_extension_dataset, 'w') as f:
            f.write('')

        with pytest.raises(ValueError):
            client.upload_file(invalid_extension_dataset, wrapper_id)

        csv_dataset = tmp_path / 'dataset.csv'
        with open(csv_dataset, 'w') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',')
            employee_writer.writerow(['Anders', 'Fabian', 'Alex'])
            employee_writer.writerow(['Martin', 'Daniel', 'Lena'])

        with ZipFile(csv_dataset, 'w') as myzip:
            myzip.write(csv_dataset)

        assert client.upload_file(csv_dataset, wrapper_id) == upload_id


class TestPolling:
    def test_ongoing(self):

        def f():
            return [
                UploadJob('1', Status.SUCCESS),
                UploadJob('2', Status.SUCCESS),
                UploadJob('3', Status.PROCESSING),
                UploadJob('4', Status.FAILED),
                UploadJob('5', Status.FAILED),

            ]

        polling = Polling(f)
        polling.update()
        assert polling.failed_jobs == ['4', '5']
        assert polling.successful_jobs == ['1', '2']
        assert polling.ongoing is True

    def test_not_ongoing(self):

        def f():
            return [UploadJob('1', Status.SUCCESS)]

        polling = Polling(f)
        polling.update()
        assert polling.successful_jobs == ['1']
        assert polling.failed_jobs == []
        assert polling.ongoing is False

    def test_invalid_status(self):

        def f():
            return [UploadJob('1', 'PENDING')]

        polling = Polling(f)
        with pytest.raises(ValueError):
            polling.update()
