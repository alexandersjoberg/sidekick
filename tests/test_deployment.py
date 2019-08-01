import random
from typing import List

import numpy as np
import pytest
import responses
from PIL import Image

import sidekick
from sidekick import Deployment
from sidekick.data_models import FeatureSpec


def get_feature(dtype: str, shape: List[int]):
    return {
        'extensions': {
            'x-peltarion': {
                'type': dtype,
                'shape': shape,
            },
        },
    }


def get_properties(features: List[FeatureSpec]) -> dict:
    return {
        feature.name: get_feature(feature.dtype, feature.shape)
        for feature in features
    }


def mock_api_specs(
    features_in: List[FeatureSpec],
    features_out: List[FeatureSpec],
) -> dict:

    return {
        'components': {
            'schemas': {
                'input-row': {
                    'properties': get_properties(features_in)
                },
                'output-row-batch': {
                    'properties': {
                        'rows': {
                            'properties': get_properties(features_out)
                        },
                    },
                },
            },
        },
    }


@responses.activate
def test_deployment_instantiation():
    features_in = [FeatureSpec('input', 'image', (100, 100, 3))]
    features_out = [FeatureSpec('output', 'numeric', (1,))]

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    assert deployment.feature_specs_in[0].name == features_in[0].name
    assert deployment.feature_specs_in[0].dtype == features_in[0].dtype
    assert deployment.feature_specs_in[0].shape == tuple(features_in[0].shape)

    assert deployment.feature_specs_out[0].name == features_out[0].name
    assert deployment.feature_specs_out[0].dtype == features_out[0].dtype
    assert (
        deployment.feature_specs_out[0].shape == tuple(features_out[0].shape)
    )

    # Ensure feature_specs cannot be modified
    deployment.feature_specs_in[0].shape = (0, 0, 0)
    deployment.feature_specs_out[0].name = 'string'
    assert deployment.feature_specs_in[0].shape == features_in[0].shape
    assert deployment.feature_specs_out[0].name == features_out[0].name


@responses.activate
def test_deployment_numeric_single_input():
    features_in = [FeatureSpec('input', 'numeric', (1,))]
    features_out = [FeatureSpec('output', 'numeric', (1,))]
    prediction = 1

    responses.add(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': prediction}]}
    )

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    predictions = deployment.predict(input=1.0)
    assert predictions == {'output': prediction}


@responses.activate
def test_deployment_numeric_multiple_input():
    features_in = [
        FeatureSpec('input_1', 'numeric', (1,)),
        FeatureSpec('input_2', 'numeric', (1,)),
    ]
    features_out = [FeatureSpec('output', 'numeric', (1,))]
    output = 1

    responses.add(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': output}]}
    )

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    # Single numeric prediction
    predictions = deployment.predict(input_1=1.0, input_2=2)
    assert predictions == {'output': output}

    inputs = [{'input_1': 1.0, 'input_2': 2} for _ in range(10)]

    # List of numeric predictions
    predictions = deployment.predict_many(inputs)
    for prediction in predictions:
        np.testing.assert_array_equal(prediction['output'], output)

    # Generator of numeric predictions
    predictions = deployment.predict_lazy(inputs)
    for prediction in predictions:
        np.testing.assert_array_equal(prediction['output'], output)

    # Incorrect type
    with pytest.raises(TypeError):
        deployment.predict(input_1=1.0, input_2='foo')


@responses.activate
def test_deployment_text():
    categories = 10
    features_in = [FeatureSpec('input', 'text', (20,))]
    features_out = [FeatureSpec('output', 'categorical', (categories,))]
    predictions = {str(i): random.random() for i in range(categories)}

    responses.add(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': predictions}]},
    )

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    prediction = deployment.predict(input='foo')
    assert prediction == {'output': predictions}


@responses.activate
def test_deployment_categorical():
    categories = 10
    features_in = [
        FeatureSpec('input_1', 'numeric', (1,)),
        FeatureSpec('input_2', 'numpy', (30,)),
    ]
    features_out = [FeatureSpec('output', 'categorical', (categories,))]
    predictions = {str(i): random.random() for i in range(categories)}

    responses.add(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': predictions}]},
    )

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    prediction = deployment.predict(input_1=1.0, input_2=np.random.rand(30))
    assert prediction == {'output': predictions}

    # Send bad type
    with pytest.raises(TypeError):
        deployment.predict(input_1=1.0, input_2='foo')

    # Return bad shape
    predictions = {str(i): random.random() for i in range(5)}
    responses.replace(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': predictions}]},
    )

    with pytest.raises(ValueError):
        deployment.predict(input_1=1.0, input_2=np.random.rand(30))


@responses.activate
def test_deployment_numpy_autoencoder():
    shape = (100, 10, 3)
    features_in = [FeatureSpec('input', 'numpy', shape)]
    features_out = [FeatureSpec('output', 'numpy', shape)]

    arr = np.random.rand(*shape).astype(np.float32)
    encoded = sidekick.encode.NumpyEncoder().encode_json(arr)

    responses.add(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': encoded}]},
    )

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    # Single numpy prediction
    prediction = deployment.predict(input=arr)
    np.testing.assert_array_equal(prediction['output'], arr)

    # List of numpy predictions
    predictions = deployment.predict_many({'input': arr} for _ in range(10))
    for prediction in predictions:
        np.testing.assert_array_equal(prediction['output'], arr)

    # Generator of numpy predictions
    predictions = deployment.predict_lazy({'input': arr} for _ in range(10))
    for prediction in predictions:
        np.testing.assert_array_equal(prediction['output'], arr)

    # Send bad shape
    with pytest.raises(ValueError):
        deployment.predict(input=np.random.rand(100, 1, 1))


@responses.activate
def test_deployment_image_autoencoder():
    shape = (100, 10, 3)
    features_in = [FeatureSpec('input', 'image', shape)]
    features_out = [FeatureSpec('output', 'image', shape)]
    arr = np.uint8(np.random.rand(*shape) * 255)
    image = Image.fromarray(arr)
    image.format = 'png'

    encoded = sidekick.encode.ImageEncoder().encode_json(image)
    responses.add(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': encoded}]}
    )

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    prediction = deployment.predict(input=image)
    np.testing.assert_array_equal(np.array(prediction['output']), arr)

    # Send bad type
    with pytest.raises(TypeError):
        deployment.predict(input=arr)


@responses.activate
def test_deployment_user_agent():
    features_in = [FeatureSpec('input', 'numeric', (1,))]
    features_out = [FeatureSpec('output', 'numeric', (1,))]
    prediction = 1

    responses.add(
        responses.POST,
        'http://peltarion.com/deployment/forward',
        json={'rows': [{'output': prediction}]}
    )

    responses.add(
        responses.GET,
        'http://peltarion.com/deployment/openapi.json',
        json=mock_api_specs(features_in, features_out),
    )

    deployment = Deployment(
        url='http://peltarion.com/deployment/forward',
        token='deployment_token',
    )

    predictions = deployment.predict(input=1.0)
    request = responses.calls[0].request
    assert predictions == {'output': prediction}
    assert len(responses.calls) == 2
    assert 'sidekick' in request.headers['User-Agent'].lower()
