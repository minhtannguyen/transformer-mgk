"""Create the feature map interface and some commonly used feature maps.

All attention implementations that expect a feature map shall receive a factory
function that returns a feature map instance when called with the query
dimensions.
"""

from functools import partial

import torch
from torch.nn import Module
import math


class FeatureMap(Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)


tanh_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.tanh(x) + 1
)

rtanh_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: 1 - torch.tanh(x)
)

elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)

elu2_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(-x) + 1
)

bao_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.where(x>=1, x+1, torch.where(x<=-1, torch.exp(x)+2-math.exp(-1), 2*torch.square(x)))
)

orthog_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.where(x>=1, torch.exp(-x)+2-math.exp(-1), torch.where(x<=-1, x-1, 2*x))
)

r_elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.where(x>0, (torch.square(x) * torch.exp(-x)), (torch.square(x) * (x-1)))
)

relu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.relu(x)
)

relu2_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.relu(-x)
)

gelu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.gelu(x) + 0.16998
)

celu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.celu(x) + 1
)

sigmoid_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.sigmoid(x)
)

leakyrelu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.leaky_relu(x) + 1
)

softplus_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.softplus(x)
)

