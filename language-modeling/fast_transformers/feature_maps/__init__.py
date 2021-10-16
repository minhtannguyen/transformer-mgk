"""Implementations of feature maps to be used with linear attention and causal
linear attention."""


from .base import tanh_feature_map, elu_feature_map, elu2_feature_map, ActivationFunctionFeatureMap
from .fourier_features import RandomFourierFeatures, Favor, \
    SmoothedRandomFourierFeatures, GeneralizedRandomFeatures
