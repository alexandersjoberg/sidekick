import pkg_resources

from . import deployment, encode, dataset_client
from .dataset import create_dataset, process_image
from .deployment import Deployment
from .dataset_client import DatasetClient

__all__ = [
    'Deployment',
    'DatasetClient',
    'create_dataset',
    'deployment',
    'dataset_client'
    'encode',
    'process_image'
]

try:
    __version__ = pkg_resources.get_distribution('sidekick').version
except pkg_resources.DistributionNotFound:
    __version__ = '0.0.0-local'
