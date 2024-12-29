# 计算边的特征

import math
import numpy as np


class EdgeFeatures:
    def __init__(self):
        pass

    # 计算两个点的中点
    @staticmethod
    def midpoint(v1, v2):
        return (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2

    # 计算位置偏移 δij = mj - mi
    def position_offset(self, vi, vj):
        mi = self.midpoint(vi[0], vi[1])
        mj = self.midpoint(vj[0], vj[1])
        return mj[0] - mi[0], mj[1] - mi[1]  # 返回x, y方向的偏移量

    # 计算方向偏移 ∡ij（两个向量的夹角）
    @staticmethod
    def direction_offset(vi, vj):
        # vi 和 vj 的方向向量
        v1 = (vi[1][0] - vi[0][0], vi[1][1] - vi[0][1])
        v2 = (vj[1][0] - vj[0][0], vj[1][1] - vj[0][1])

        # 计算向量之间的夹角（余弦定理）
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # 保证数值在[-1, 1]范围内，避免浮动误差
        cos_angle = max(-1, min(1, cos_angle))

        # 计算夹角的弧度，再转为角度
        angle = math.acos(cos_angle)  # 结果以弧度为单位
        return angle  # 返回弧度

    # 计算顶点 vi 的长度
    @staticmethod
    def length(vi):
        return math.sqrt((vi[1][0] - vi[0][0]) ** 2 + (vi[1][1] - vi[0][1]) ** 2)

    # 计算长度比 rij = li / (li + lj)
    def length_ratio(self, vi, vj):
        li = self.length(vi)
        lj = self.length(vj)
        return li / (li + lj)  # 长度比

    # 计算平行性
    @staticmethod
    def is_parallel(vi, vj):

        # 计算两个向量的叉积为0时表示平行
        v1 = (vi[1][0] - vi[0][0], vi[1][1] - vi[0][1])
        v2 = (vj[1][0] - vj[0][0], vj[1][1] - vj[0][1])

        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return abs(cross_product) < 1e-6  # 近似为零表示平行

    # 计算正交性
    @staticmethod
    def is_orthogonal(vi, vj):
        v1 = (vi[1][0] - vi[0][0], vi[1][1] - vi[0][1])
        v2 = (vj[1][0] - vj[0][0], vj[1][1] - vj[0][1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        return abs(dot_product) < 1e-6  # 近似为零表示正交

    # 检查是否共享相同的端点
    @staticmethod
    def shares_endpoint(vi, vj):
        return vi[0] == vj[0] or vi[1] == vj[1]  # 检查端点是否相同

    # 计算边的特征
    def compute_edge_features(self, vi, vj):
        # δij = mj - mi
        position_offset = self.position_offset(vi, vj)

        # ∡ij = 锐角
        direction_offset = self.direction_offset(vi, vj)

        # rij = li / (li + lj)
        rij = self.length_ratio(vi, vj)

        # 计算平行、正交和共享端点
        parallel = 1 if self.is_parallel(vi, vj) else 0
        orthogonal = 1 if self.is_orthogonal(vi, vj) else 0
        shares_endpoint = 1 if self.shares_endpoint(vi, vj) else 0

        # 二元指示器列表
        gij = [parallel, orthogonal, shares_endpoint]

        # 返回一个包含所有特征的数组
        return np.array([position_offset[0], position_offset[1], direction_offset, rij] + gij)


# 示例：计算边特征
# vi = [(1, 1), (2, 3)]  # 顶点 vi
# vj = [(4, 5), (6, 7)]  # 顶点 vj
#
# # 创建 EdgeFeatures 类的实例
# edge_feature_calculator = EdgeFeatures()
#
# # 计算边特征
# features = edge_feature_calculator.compute_edge_features(vi, vj)
# print("Edge features:", features)
