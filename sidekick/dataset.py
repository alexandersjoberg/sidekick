import collections
import functools
import multiprocessing
import os
from typing import Any, Callable, Iterable, Mapping, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .encode import ENCODER_COMPATIBILITY, FILE_EXTENSION_ENCODERS


def crop_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    width, height = size
    center_x, center_y = (i // 2 for i in image.size)
    diff_x, diff_y = width // 2, height // 2

    left = center_x - diff_x
    right = center_x + diff_x + width % 2

    lower = center_y - diff_y
    upper = center_y + diff_y + height % 2

    cropped_image = image.crop((left, lower, right, upper))
    cropped_image.load()
    cropped_image.format = image.format
    return cropped_image


def resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    resized_image = image.resize(size)
    resized_image.format = image.format
    return resized_image


def process_image(image: Image.Image,
                  mode: str = 'center_crop_or_pad',
                  size: Tuple[int, int] = None,
                  file_format: str = None):
    """Process image
    Args:
        image: Image to process (will be modified)
        mode: {'center_crop_or_pad', 'resize'}.
              Only active if size is not None.
        size: Output image size (width, height)
        file_format: Set to modify format. Can be 'png' or 'jpeg'.
    Returns:
        image
    """
    modes = {'center_crop_or_pad', 'resize'}

    if size:
        if mode == 'center_crop_or_pad':
            image = crop_image(image, size)

        elif mode == 'resize':
            image = resize_image(image, size)

        else:
            raise ValueError('Mode not supported. Available: %s' % modes)

    if file_format is not None:
        if file_format not in FILE_EXTENSION_ENCODERS:
            raise ValueError('Not supported format: ' + file_format)
        image.format = file_format

    return image


def create_dataset(dataset_path: str,
                   dataset_index: pd.DataFrame,
                   path_columns: Iterable[str] = None,
                   preprocess: Mapping[str, Callable] = None,
                   include_index: bool = False,
                   parallel_processing: int = 10,
                   progress: bool = False,
                   overwrite: bool = False):
    """Create a Peltarion compatible .zip dataset

    Notice that columns containing images must have the same shape. Please use
    the `process_image` preprocessor to ensure they are all the same shape if
    `verify_images` states there are differances.

    Args:
        dataset_path: Path (including .zip extension) to dataset
        dataset_index: DataFrame with data to encode
        path_columns: Columns in the index file which correspond to paths to be
                      loaded into the zipfile
        preprocess: Maps column names to preprocessing functions
        include_index: Include the dataset index (creates new field with index)
        parallel_processing: How many processes to parallel to the existing
                             process for preprocessing. Set to 0 to disable.
        progress: Print progress
        overwrite: Overwrite the output file if exists, otherwise exit with
                   an exception

    See Also:
        verify_images: To verify image columns are platform compatible
        process_image: To process all images to a platform compatible format
    """
    # Sanity checks
    if os.path.exists(dataset_path):
        if overwrite:
            os.remove(dataset_path)
        else:
            raise OSError("File %s already exists, will not overwrite" %
                          dataset_path)
    if not len(dataset_index):
        raise ValueError('Empty dataset index')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    object_columns = {}
    for column, dtype in dataset_index.dtypes.items():
        if (dtype == np.dtype(object) and
                not isinstance(dataset_index[column].iloc[0], str)):
            object_columns[column] = type(dataset_index[column].iloc[0])

    unsupported_types = set(object_columns.values()).difference(
        ENCODER_COMPATIBILITY)
    if unsupported_types:
        raise TypeError('Unsupported types encountered: %s' %
                        unsupported_types)

    # Fix defaults and copy dataframe to avoid modifying users copy
    dataset_index = dataset_index.copy()
    if preprocess is None:
        preprocess = dict()
    path_columns = set(path_columns if path_columns is not None else [])
    process_columns = path_columns.union(preprocess).union(object_columns)
    status_bar = tqdm(
        total=len(dataset_index) * len(process_columns) + 1,
        disable=not progress
    )
    with ZipFile(dataset_path, 'w', compression=ZIP_DEFLATED) as dataset_zip:
        # Add metadata file to track usage
        dataset_zip.writestr('metadata.json', '{ "source" : "sidekick" }')

        # Copy over without preprocessing
        for column in path_columns.difference(preprocess):
            for index, item in dataset_index[column].items():
                relative_path = os.path.join(
                    column, str(index) + os.path.splitext(item)[1])
                dataset_zip.write(item, relative_path)
                dataset_index.at[index, column] = relative_path
                status_bar.update()

        # Copy over items requiring preprocessing or encoding
        process_columns = set(preprocess).union(object_columns)
        rows = dataset_index[list(process_columns)].iterrows()
        preprocessing_fun = functools.partial(
            _preprocess, path_columns=path_columns, preprocess=preprocess)
        if parallel_processing > 0:
            with multiprocessing.Pool(processes=parallel_processing) as pool:
                rows = pool.imap_unordered(preprocessing_fun, rows)
                _store_preprocessed_rows(
                    dataset_zip, dataset_index, rows, status_bar.update)
        else:
            rows = (preprocessing_fun(row) for row in rows)
            _store_preprocessed_rows(
                dataset_zip, dataset_index, rows, status_bar.update)

        # Write index file
        content = dataset_index.to_csv(index=include_index)
        dataset_zip.writestr('index.csv', content)
        status_bar.update()
    status_bar.close()


_Preprocessed = collections.namedtuple(
    '_Preprocessed', ['index', 'paths', 'files'])


def _preprocess(index_row_pair: Tuple[int, pd.Series],
                path_columns: Iterable[str],
                preprocess: Mapping[str, Callable[[Any], Any]]) \
        -> _Preprocessed:
    """Preprocess a row of a dataset

    Columns that are in `path_columns` will be loaded from disk. Those which
    are listed in `preprocess` will processed by the provided callable before
    being encoded to a binary and returned.

    Args:
        index_row_pair: tuple of index for row and row data
        path_columns: columns with paths to load from disk
        preprocess: maps column names to callables which accepts and processes
                    the decoded type

    Returns:
        Preprocessed data
    """
    index, row = index_row_pair
    processed = dict()
    paths = dict()
    for key, value in row.items():
        # Load files
        if key in path_columns:
            basename = os.path.basename(value)
            _, file_extension = basename.rsplit('.', 1)
            encoder = FILE_EXTENSION_ENCODERS[file_extension]
            with open(value, 'rb') as f:
                value = encoder.decode(f.read())
        else:
            encoder = ENCODER_COMPATIBILITY[type(value)]

        # TODO: Allow preprocessor to change type by looking up encoder after
        # Do preprocessing
        if key in preprocess:
            value = preprocess[key](value)

        # Encode, determine filename and extension
        file_extension = encoder.file_extension(value)
        filename = index
        relative_path = os.path.join(key, '%s.%s' % (filename, file_extension))
        processed[relative_path] = encoder.encode(value)
        paths[key] = relative_path
    return _Preprocessed(index, paths, processed)


def _store_preprocessed_rows(zf: ZipFile,
                             dataset_index: pd.DataFrame,
                             preprocessed: Iterable[_Preprocessed],
                             callback: Callable):
    """Store preprocessed items in zip and update dataset index

    Args:
        zf: zipfile to write items to
        dataset_index: dataset index
        preprocessed: preprocessed items
        callback: callback to run after writing to zipfile, e.g. for prog. bar
    """
    for index, paths, processed in preprocessed:
        for key, value in processed.items():
            zf.writestr(key, value)
            callback()
        for key, path in paths.items():
            dataset_index.at[index, key] = path
