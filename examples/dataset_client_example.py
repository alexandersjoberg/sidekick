import concurrent.futures
import time
from functools import partial
from typing import List

from sidekick.dataset_client import DatasetClient, Polling


def create_dataset_and_upload_many_files(
    file_paths: List[str],
    url: str,
    token: str,
    dataset_name: str = 'Sidekick upload',
    dataset_description: str = 'Sidekick upload',
) -> None:

    client = DatasetClient(url, token)
    wrapper_id = client.create_wrapper(dataset_name, dataset_description)

    workers = min(10, len(file_paths))
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
