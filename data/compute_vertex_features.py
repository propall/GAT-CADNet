"""
用于计算图中每个顶点的特征
vertex_feature = [cos(2a_i), sin(2a_i), l_i, t_i]
其中a是坐标原点到顶点的连线与x轴的夹角
l_i: sqrt(x^2 + y^2)
t_i: 针对弧线、线段、圆圈以及椭圆的one-hot编码
"""

import numpy as np


# 计算角度和长度
def compute_angle_and_length(x, y):
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += np.pi  # 保证角度在[0, pi)之间
    # 计算长度
    length = np.sqrt(x**2 + y**2)
    return angle, length


# 计算方向特征
def compute_direction_features(angle):
    return np.cos(2 * angle), np.sin(2 * angle)


# 创建one-hot编码(类型)
def one_hot_coding(segment_type):
    types = ['line', 'arc', 'circle', 'ellipse']
    one_hot = np.zeros(4)
    one_hot[types.index(segment_type)] = 1
    return one_hot


# 计算顶点特征
def compute_vertex_features(vertices, segment_types):
    features = []
    for (x, y), segment_type in zip(vertices, segment_types):
        # 计算角度和长度
        angle, length = compute_angle_and_length(x, y)
        # 计算方向特征
        cos_angle, sin_angle = compute_direction_features(angle)
        # 创建one-hot编码
        type_encoding = one_hot_coding(segment_type)
        # 将特征组合成一个向量
        feature = [cos_angle, sin_angle, length] + type_encoding.tolist()
        features.append(feature)
    return np.array(features)
