import functools
import os
import sys
import tempfile
from random import random
from typing import Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

import sidekick


def balance_dataset(dataset: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Balances dataset by oversampling under-represented classes."""
    class_count = dataset[target_column].value_counts().to_dict()
    largest_class = max(class_count, key=class_count.get)
    counts_largest_class = class_count[largest_class]

    for class_name in class_count:
        samples = counts_largest_class - class_count[class_name]
        sampling = dataset[dataset[target_column] == class_name].sample(
            samples, replace=True)
        dataset = pd.concat([dataset, sampling], ignore_index=True)

    return dataset.sample(frac=1).reset_index(drop=True)


def create_ham_dataset(
        directory: str = None,
        size: Tuple[int, int] = (224, 224),
        split: float = 0.8,
        balance: bool = True,
) -> None:
    """Creates a Peltarion-compatible zip file with the HAM10000 dataset.

    The HAM10000 dataset contains labeled images of different types of skin
    lesions. Read more here: https://arxiv.org/abs/1803.10417.

    All data is provided under the terms of the Creative Commons
    Attribution-NonCommercial (CC BY-NC) 4.0 license. You may find the terms of
    the licence here: https://creativecommons.org/licenses/by-nc/4.0/legalcode.
    If you are unable to accept the terms of this license, do not download or
    use this data.

    Please notice that the disclaimer in the README.md applies.

    Args:
        directory: Directory where the dataset will be stored. If not provided,
            it defaults to the current working directory.
        size: Image size after resizing: (width, height). The original image
            size is (600, 450).
        split: Split fraction between training and validation.
        balance: Balance training dataset by oversampling.
    """""

    images_dir = 'ISIC2018_Task3_Training_Input'
    metadata_dir = 'ISIC2018_Task3_Training_GroundTruth'
    metadata_file = 'ISIC2018_Task3_Training_GroundTruth.csv'

    metadata_url = 'https://challenge.kitware.com/api/v1/item/' \
                   '5ac20eeb56357d4ff856e136/download'
    images_url = 'https://challenge.kitware.com/api/v1/item/' \
                 '5ac20fc456357d4ff856e139/download'

    if directory is None:
        directory = os.getcwd()

    if not os.path.isdir(directory):
        sys.exit('Directory provided does not exist')

    dataset_path = os.path.join(directory, 'ham_dataset.zip')
    with tempfile.TemporaryDirectory() as tmpdir:

        print('Downloading metadata...')
        metadata, _ = urlretrieve(metadata_url)
        with ZipFile(metadata) as zip_handle:
            zip_handle.extractall(tmpdir)

        print('Downloading images. This step might take some time.')
        images, _ = urlretrieve(images_url)
        with ZipFile(images) as zip_handle:
            zip_handle.extractall(tmpdir)

        # read metadata
        df = pd.read_csv(os.path.join(tmpdir, metadata_dir, metadata_file))

        # decode one-hot encoding
        categories = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        df['target'] = df[categories].idxmax(axis=1)
        df = df.drop(categories, axis=1)

        # split dataset into train and validation
        rows = df.shape[0]
        subset = ['train' if random() < split else 'val' for _ in range(rows)]
        df['subset'] = subset

        if balance:
            train = df[df['subset'] == 'train']
            val = df[df['subset'] == 'val']
            train_balanced = balance_dataset(train, 'target')
            df = pd.concat([train_balanced, val], ignore_index=True)

        # replace image name by image path
        df['image'] = df['image'].apply(
            lambda x: os.path.join(tmpdir, images_dir, x + '.jpg'))

        image_processor = functools.partial(
            sidekick.process_image,
            mode='resize',
            size=size,
            file_format='jpeg'
        )

        print('Creating dataset...')
        sidekick.create_dataset(
            dataset_path,
            df,
            path_columns=['image'],
            preprocess={'image': image_processor},
            progress=True,
        )


if __name__ == '__main__':
    create_ham_dataset()
