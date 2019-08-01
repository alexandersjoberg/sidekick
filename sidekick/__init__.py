import pkg_resources

from . import deployment, encode
from .dataset import create_dataset, process_image
from .deployment import Deployment

__all__ = [
    'Deployment',
    'create_dataset',
    'deployment',
    'encode',
    'process_image'
]

try:
    __version__ = pkg_resources.get_distribution('sidekick').version
except pkg_resources.DistributionNotFound:
    __version__ = '0.0.0-local'
