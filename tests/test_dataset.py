import functools
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import sidekick


@pytest.fixture
def dataset_index(tmpdir):
    n_rows = 32

    # Create image stored in temporary file
    image_file_path = str(tmpdir.join('test_image.jpg'))
    Image.new(mode='RGB', size=(640, 320)).save(image_file_path)

    # Create columns for dataset
    integer_column = np.random.randint(0, 10, size=n_rows)
    float_column = np.random.rand(n_rows)
    string_column = ['foo'] * n_rows
    numpy_column = list(np.random.rand(n_rows, 3))
    image_column = [
        Image.new(mode='RGB', size=(64, 32)) for _ in range(n_rows)
    ]
    image_file_column = [image_file_path for _ in range(n_rows)]

    # Build dataset index
    return pd.DataFrame({
        'integer_column': integer_column,
        'float_column': float_column,
        'string_column': string_column,
        'numpy_column': numpy_column,
        'image_column': image_column,
        'image_file_column': image_file_column,
        'image_file_process_column': image_file_column
    })


def test_create_dataset_sequential(dataset_index, tmpdir):
    # Create dataset
    dataset_path = str(tmpdir.join('dataset.zip'))
    crop_image = functools.partial(
        sidekick.process_image,
        mode='center_crop_or_pad',
        size=(32, 8),
        file_format='png'
    )
    set_image_format = functools.partial(
        sidekick.process_image, file_format='png')

    sidekick.create_dataset(
        dataset_path,
        dataset_index,
        path_columns=['image_file_column', 'image_file_process_column'],
        preprocess={
            'image_file_process_column': crop_image,
            'image_column': set_image_format
        },
        progress=True,
        parallel_processing=0
    )
    assert os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 100


def test_create_dataset_parallel(dataset_index, tmpdir):
    # Create dataset
    dataset_path = str(tmpdir.join('dataset.zip'))
    resize_image = functools.partial(
        sidekick.process_image,
        mode='resize',
        size=(32, 8),
        file_format='png'
    )
    set_image_format = functools.partial(
        sidekick.process_image, file_format='png')

    sidekick.create_dataset(
        dataset_path,
        dataset_index,
        path_columns=['image_file_column', 'image_file_process_column'],
        preprocess={
            'image_file_process_column': resize_image,
            'image_column': set_image_format
        },
        progress=False,
        parallel_processing=10
    )
    assert os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 100


def test_dataset_metadata(dataset_index, tmpdir):
    # Create dataset
    dataset_path = str(tmpdir.join('dataset.zip'))
    set_image_format = functools.partial(
        sidekick.process_image, file_format='png')
    sidekick.create_dataset(
        dataset_path,
        dataset_index,
        preprocess={'image_column': set_image_format}
    )

    # Assert that the metadata file was added
    with zipfile.ZipFile(dataset_path, 'r') as zf:
        metadata = zf.read('metadata.json')
        assert metadata == b'{ "source" : "sidekick" }'


def test_import_multiple_formats(tmpdir):
    size = (64, 32)
    images = [
        Image.new(mode='RGBA', size=size),
        Image.new(mode='LA', size=size),
        Image.new(mode='RGB', size=size),
        Image.new(mode='L', size=size)
    ]

    df = pd.DataFrame({
        'image_column': images
    })
    dataset_path = str(tmpdir.join('dataset.zip'))
    set_image_format = functools.partial(
        sidekick.process_image, file_format='png')

    sidekick.create_dataset(
        dataset_path,
        df,
        preprocess={
            'image_column': set_image_format
        },
        progress=True,
        parallel_processing=0
    )
    assert os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 100


def test_process_image_modes():
    target_size = (299, 399)
    file_format = 'png'
    image = Image.new(mode='RGB', size=(600, 450))
    modes = ('center_crop_or_pad', 'crop_and_resize', 'resize')
    for mode in modes:
        new_image = sidekick.process_image(
            image=image,
            mode=mode,
            size=target_size,
            file_format=file_format
        )
        assert new_image.size == target_size
        assert new_image.format == file_format


def test_crop_image():
    size = (10, 15)
    file_format = 'png'
    image = Image.new(mode='RGB', size=(100, 100))
    image.format = file_format
    cropped_image = sidekick.dataset.crop_image(image, size=size)
    assert cropped_image.size == size
    assert cropped_image.format == file_format


def test_crop_and_resize_image():
    # Assert nothing happens if size matches
    size = (10, 10)
    file_format = 'png'
    image = Image.fromarray(np.random.rand(*size))
    image.format = file_format
    new_image = sidekick.dataset.crop_and_resize_image(
        image=image, size=size)

    assert new_image.size == size
    assert new_image.format == file_format
    assert np.allclose(np.array(image), np.array(new_image))

    # Assert that there is only a resize if proportions match
    new_size = (20, 20)
    new_image = sidekick.dataset.crop_and_resize_image(
        image=image, size=new_size)

    assert new_image.size == new_size
    assert new_image.format == file_format
    assert np.allclose(np.array(image.resize(new_size)), np.array(new_image))

    # Assert can handle change of proportion and size
    new_size = (20, 10)
    new_image = sidekick.dataset.crop_and_resize_image(
        image=image, size=new_size)

    assert new_image.format == file_format
    assert new_image.size == new_size


def test_resize_image():
    size = (10, 15)
    file_format = 'png'
    image = Image.new(mode='RGB', size=(100, 100))
    image.format = file_format
    resized_image = sidekick.dataset.resize_image(image, size=size)
    assert resized_image.size == size
    assert resized_image.format == file_format
