import copy
import urllib.parse
from itertools import islice
from typing import Any, Dict, Generator, Iterable, List

import requests
from requests.adapters import HTTPAdapter

from .data_models import FeatureSpec
from .encode import DataItem, decode_feature, encode_feature

PredictData = Dict[str, List[Dict[str, Any]]]


def prediction_request(items: Iterable[DataItem],
                       feature_specs: List[FeatureSpec]) -> PredictData:
    rows = list()
    for item in items:
        row = dict()
        for feature_spec in feature_specs:
            if feature_spec.name not in item:
                raise ValueError(
                    'Item is missing feature: %s' % feature_spec.name
                )
            feature = item[feature_spec.name]
            row[feature_spec.name] = encode_feature(feature, feature_spec)
        rows.append(row)
    return {'rows': rows}


def parse_prediction(
        data: PredictData,
        feature_specs: List[FeatureSpec]) -> Generator[DataItem, None, None]:
    if 'errorCode' in data:
        raise IOError(
            '%s: %s' % (data['errorCode'], data.get('errorMessage', '')))
    if 'rows' not in data:
        raise ValueError('Return data does not contain rows')

    for row in data['rows']:
        item = dict()
        for feature_spec in feature_specs:
            if feature_spec.name not in row:
                raise ValueError(
                    'Item is missing feature: %s' % feature_spec.name
                )

            item[feature_spec.name] = decode_feature(
                row[feature_spec.name],
                feature_spec,
            )
        yield item


def get_feature_specs(specs: Dict) -> List[FeatureSpec]:
    return [
        FeatureSpec(
            name=feature_name,
            dtype=specs['extensions']['x-peltarion']['type'],
            shape=tuple(specs['extensions']['x-peltarion']['shape']),
        )
        for feature_name, specs in specs.items()
    ]


class Deployment:
    """Sidekick for Peltarion platform deployments
    """
    BATCH_SIZE = 128
    MAX_RETRIES = 3

    def __init__(self, url: str, token: str) -> None:
        self._headers = {'Authorization': 'Bearer ' + token}
        self._url = url

        self._session = requests.Session()
        self._session.mount('', HTTPAdapter(max_retries=self.MAX_RETRIES))
        self._session.headers.update({'User-Agent': 'sidekick'})

        response = self._session.get(
            url=urllib.parse.urljoin(url, 'openapi.json'),
            headers=self._headers,
        )
        response.raise_for_status()
        specs = response.json()['components']['schemas']
        self._feature_specs_in = get_feature_specs(
            specs['input-row']['properties']
        )
        self._feature_specs_out = get_feature_specs(
            specs['output-row-batch']['properties']['rows']['properties']
        )

    @property
    def feature_specs_in(self) -> List[FeatureSpec]:
        return copy.deepcopy(self._feature_specs_in)

    @property
    def feature_specs_out(self) -> List[FeatureSpec]:
        return copy.deepcopy(self._feature_specs_out)

    def predict_lazy(self, items: Iterable[DataItem]) -> \
            Generator[DataItem, None, None]:
        iterator = iter(items)
        while True:
            batch = list(islice(iterator, self.BATCH_SIZE))
            if not batch:
                break

            encoded = prediction_request(batch, self._feature_specs_in)
            response = self._session.post(
                url=self._url,
                headers=self._headers,
                json=encoded
            )
            response.raise_for_status()  # Raise exceptions
            yield from parse_prediction(
                response.json(),
                self._feature_specs_out
            )

    def predict_many(self, items: Iterable[DataItem]) -> List[DataItem]:
        return list(self.predict_lazy(items))

    def predict(self, **item) -> DataItem:
        return self.predict_many([item])[0]
