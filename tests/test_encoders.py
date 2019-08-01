import numpy as np
import pytest
from PIL import Image

from sidekick.encode import (CategoricalEncoder, ImageEncoder, NumericEncoder,
                             NumpyEncoder, TextEncoder)


def test_numeric_encoder():
    encoder = NumericEncoder()

    encoder.check_type(2)
    encoder.check_type(2.1)

    with pytest.raises(TypeError):
        encoder.check_type('string')


def test_text_encoder():
    encoder = TextEncoder()
    value = 'string'

    encoder.check_type(value)
    encoder.check_shape(value, shape=(100,))

    with pytest.raises(TypeError):
        encoder.check_type(34)

    with pytest.warns(UserWarning):
        encoder.check_shape(value, shape=(2,))


def test_categorical_encoder():
    encoder = CategoricalEncoder()
    value = {'a': 1, 'b': 2}

    encoder.check_type(value)
    with pytest.raises(TypeError):
        encoder.check_type(5)

    encoder.check_shape(value, (2,))
    with pytest.raises(ValueError):
        encoder.check_shape(value, (3,))


def test_image_encoder():
    encoder = ImageEncoder()

    shape = (100, 10, 3)
    arr = np.uint8(np.random.rand(*shape) * 255)
    image = Image.fromarray(arr)

    encoder.check_type(image)
    with pytest.raises(TypeError):
        encoder.check_type(arr)

    with pytest.raises(ValueError):
        encoder.file_extension(image)

    image.format = 'PNG'
    assert encoder.file_extension(image) == 'png'
    assert encoder.media_type(image) == 'image/png'

    encoder.check_shape(image, shape)
    with pytest.raises(ValueError):
        encoder.check_shape(image, (1, 10, 3))


def test_numpy_encoder():
    encoder = NumpyEncoder()

    shape = (100, 10, 3)
    arr = np.random.rand(*shape).astype(np.float32)

    encoder.check_shape(arr, shape)
    with pytest.raises(ValueError):
        encoder.check_shape(arr, (99, 10, 3))

    encoder.check_type(arr)
    with pytest.raises(TypeError):
        encoder.check_type([1, 2, 3])
