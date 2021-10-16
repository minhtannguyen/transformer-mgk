"""Implementations of feature maps to be used with linear attention and causal
linear attention."""


from .base import tanh_feature_map, rtanh_feature_map, r_elu_feature_map, elu_feature_map, elu2_feature_map, relu_feature_map, relu2_feature_map,\
celu_feature_map, sigmoid_feature_map, bao_feature_map, orthog_feature_map, \
leakyrelu_feature_map, softplus_feature_map, gelu_feature_map, ActivationFunctionFeatureMap
from .fourier_features import RandomFourierFeatures, Favor, \
    SmoothedRandomFourierFeatures, GeneralizedRandomFeatures
