from typing import Tuple


class FeatureSpec:
    def __init__(self, name: str, dtype: str, shape: Tuple[int, ...]):
        self.name = name
        self.dtype = dtype
        self.shape = shape

    def __repr__(self):
        return (
            'FeatureSpec(name="%s", dtype="%s", shape=%s)'
            % (self.name, self.dtype, self.shape)
        )
