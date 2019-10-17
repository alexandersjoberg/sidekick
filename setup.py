from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'pillow',
    'requests',
    'tqdm'
]

TEST_REQUIRED_PACKAGES = [
    'responses',
    'pytest'
]

setup(
    name='sidekick',
    version='0.2.1',
    install_requires=REQUIRED_PACKAGES,
    extras_require={'test': TEST_REQUIRED_PACKAGES},
    packages=find_packages(include='sidekick.*'),
    description='Sidekick for the Peltarion platform',
    author='Peltarion',
    author_email='support@peltarion.com',
    maintainer='alvaro-garcia-carrasco'
)
