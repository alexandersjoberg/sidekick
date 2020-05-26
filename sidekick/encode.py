import abc
import base64
import io
import itertools
from typing import Any, Mapping, Set, Tuple

import numpy as np
from PIL import Image

from .data_models import FeatureSpec

DataItem = Mapping[str, Any]


class Encoder(abc.ABC):

    def file_extension(self, value):
        _ = value
        return ''

    @abc.abstractmethod
    def expects(self):
        pass

    def encode_json(self, value):
        return self.encode(value)

    def decode_json(self, encoded):
        return self.decode(encoded)

    @abc.abstractmethod
    def encode(self, value):
        pass

    @abc.abstractmethod
    def decode(self, encoded):
        pass

    @abc.abstractmethod
    def check_shape(self, value, shape):
        pass

    def check_type(self, value):
        expected_types = self.expects()
        if not isinstance(value, tuple(expected_types)):
            raise TypeError(
                'Expected %s but received %s' % (self.expects(), type(value))
            )


class BinaryEncoder(Encoder):

    @abc.abstractmethod
    def media_type(self, value) -> str:
        pass

    @abc.abstractmethod
    def encode(self, value) -> bytes:
        pass

    @abc.abstractmethod
    def decode(self, encoded: bytes):
        pass

    def encode_json(self, value) -> str:
        return 'data:%s;base64,%s' % (
            self.media_type(value),
            base64.b64encode(self.encode(value)).decode()
        )

    def decode_json(self, encoded: str) -> Any:
        try:
            data_type, b64_data = encoded.split(',', 1)
            _, media_description = data_type.split(':', 1)
            media_type, _ = media_description.split(';', 1)
        except ValueError:
            raise ValueError('Not a valid Data URL')
        data = base64.b64decode(b64_data)
        value = self.decode(data)
        expected_media_type = self.media_type(value)
        if media_type != expected_media_type:
            raise ValueError('Not a valid media type, expected %s but got %s' %
                             (expected_media_type, media_type))
        return value


class CategoricalEncoder(Encoder):

    def expects(self) -> Set:
        return {dict}

    def check_shape(self, value: dict, shape: Tuple[int]):
        if len(value) != shape[0]:
            raise ValueError('Categorical expected %i values, got: %i'
                             % (shape[0], len(value)))

    def encode(self, value):
        return value

    def decode(self, encoded):
        return encoded


class TextEncoder(Encoder):

    def expects(self) -> Set:
        return {str}

    def check_shape(self, value: str, shape: Tuple[int]):
        pass

    def encode(self, value):
        return value

    def decode(self, encoded):
        return encoded


class NumericEncoder(Encoder):

    def expects(self) -> Set:
        return {int, float}

    def check_shape(self, value, shape):
        pass

    def encode(self, value):
        return value

    def decode(self, encoded):
        return encoded


class NumpyEncoder(BinaryEncoder):

    def file_extension(self, value):
        return 'npy'

    def media_type(self, value):
        return 'application/x.peltarion.npy'

    def expects(self) -> Set:
        return {np.ndarray}

    def check_shape(self, value: np.ndarray, shape: Tuple[int, ...]):
        if value.shape != shape:
            raise ValueError('Expected shape: %s, numpy array has shape: %s'
                             % (shape, value.shape))

    def encode(self, value: np.ndarray) -> bytes:
        value = value.astype(np.float32)
        with io.BytesIO() as buffer:
            np.save(buffer, value)
            return buffer.getvalue()

    def decode(self, encoded: bytes) -> np.ndarray:
        with io.BytesIO(encoded) as buffer:
            array = np.load(buffer)
        return array


class ImageEncoder(BinaryEncoder):

    def file_extension(self, value):
        if value.format is None:
            raise ValueError('No format set on image, please specify '
                             '(see the documentation for details)')
        return value.format.lower()

    def media_type(self, value):
        file_extension = self.file_extension(value)
        return 'image/' + file_extension

    def expects(self) -> Set:
        return {Image.Image}

    def check_shape(self, image: Image, shape: Tuple[int, ...]):
        pass

    def encode(self, value: Image) -> bytes:
        original_format = value.format
        # We do not support 4-channel PNGs or alpha in general
        if value.mode == 'RGBA':
            value = value.convert('RGB')
        elif value.mode == 'LA':
            value = value.convert('L')
        with io.BytesIO() as image_bytes:
            value.save(image_bytes, format=original_format)
            return image_bytes.getvalue()

    def decode(self, encoded: bytes):
        with io.BytesIO(encoded) as buffer:
            image = Image.open(buffer)
            image.load()
        return image


ENCODERS = {
    'numeric': NumericEncoder(),
    'categorical': CategoricalEncoder(),
    'numpy': NumpyEncoder(),
    'image': ImageEncoder(),
    'text': TextEncoder(),
}


ENCODER_COMPATIBILITY = dict(itertools.chain.from_iterable(
    ((compatible_type, encoder) for compatible_type in encoder.expects())
    for encoder in ENCODERS.values()
))


FILE_EXTENSION_ENCODERS = {
    'npy': ENCODERS['numpy'],
    'png': ENCODERS['image'],
    'jpg': ENCODERS['image'],
    'jpeg': ENCODERS['image']
}


def get_encoder(dtype: str, shape: Tuple[int, ...]) -> Encoder:
    if dtype == 'numeric' and (len(shape) > 1 or shape[0] > 1):
        return ENCODERS['numpy']
    return ENCODERS[dtype]


def encode_feature(feature, specs: FeatureSpec) -> Any:
    encoder = get_encoder(specs.dtype, specs.shape)
    encoder.check_type(feature)
    encoder.check_shape(feature, specs.shape)
    return encoder.encode_json(feature)


def decode_feature(feature, specs: FeatureSpec) -> Any:
    encoder = get_encoder(specs.dtype, specs.shape)
    decoded = encoder.decode_json(feature)
    encoder.check_type(decoded)
    encoder.check_shape(decoded, specs.shape)
    return decoded
