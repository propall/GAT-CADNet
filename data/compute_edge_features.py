# 计算边的特征

import math
import numpy as np


class EdgeFeatures:
    def __init__(self):
        pass

    # Calculate the midpoint of two points
    @staticmethod
    def midpoint(v1, v2):
        return (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2

    # Calculate position offset δij = mj - mi
    def position_offset(self, vi, vj):
        mi = self.midpoint(vi[0], vi[1])
        mj = self.midpoint(vj[0], vj[1])
        return mj[0] - mi[0], mj[1] - mi[1]  # Returns the offset in the x and y directions

    # Calculate direction offset ∡ij（The angle between two vectors）
    @staticmethod
    def direction_offset(vi, vj):
        # vi and vj Direction vector
        v1 = (vi[1][0] - vi[0][0], vi[1][1] - vi[0][1])
        v2 = (vj[1][0] - vj[0][0], vj[1][1] - vj[0][1])

        # Calculate the angle between vectors (cosine theorem)
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

        # Ensure that the value is within the range of [-1, 1] to avoid floating errors
        cos_angle = max(-1, min(1, cos_angle))

        # Calculate the radian of the included angle and then turn it into angle
        angle = math.acos(cos_angle)  # The result is in radians
        return angle  # Return to radians

    # Calculate the length of the vertex vi
    @staticmethod
    def length(vi):
        return math.sqrt((vi[1][0] - vi[0][0]) ** 2 + (vi[1][1] - vi[0][1]) ** 2)

    # Calculate the length ratio rij = li / (li + lj)
    def length_ratio(self, vi, vj):
        li = self.length(vi)
        lj = self.length(vj)
        return li / (li + lj)  # Length ratio

    # Calculate parallelism
    @staticmethod
    def is_parallel(vi, vj):

        # Calculate the cross product of two vectors to be 0 to represent parallel
        v1 = (vi[1][0] - vi[0][0], vi[1][1] - vi[0][1])
        v2 = (vj[1][0] - vj[0][0], vj[1][1] - vj[0][1])

        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return abs(cross_product) < 1e-6  # Approximately zero means parallel

    # Calculate orthogonality
    @staticmethod
    def is_orthogonal(vi, vj):
        v1 = (vi[1][0] - vi[0][0], vi[1][1] - vi[0][1])
        v2 = (vj[1][0] - vj[0][0], vj[1][1] - vj[0][1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        return abs(dot_product) < 1e-6  # Approximately zero means orthogonal

    # Check if the same endpoint is shared
    @staticmethod
    def shares_endpoint(vi, vj):
        return vi[0] == vj[0] or vi[1] == vj[1]  # Check if the endpoint is the same

    # Calculate the features of edges
    def compute_edge_features(self, vi, vj):
        # δij = mj - mi
        position_offset = self.position_offset(vi, vj)

        # ∡ij = Acute angle
        direction_offset = self.direction_offset(vi, vj)

        # rij = li / (li + lj)
        rij = self.length_ratio(vi, vj)

        # Compute parallel, orthogonal, and shared endpoints
        parallel = 1 if self.is_parallel(vi, vj) else 0
        orthogonal = 1 if self.is_orthogonal(vi, vj) else 0
        shares_endpoint = 1 if self.shares_endpoint(vi, vj) else 0

        # List of binary indicators
        gij = [parallel, orthogonal, shares_endpoint]

        # Returns an array containing all features
        return np.array([position_offset[0], position_offset[1], direction_offset, rij] + gij)


# Example: Calculate edge features
# vi = [(1, 1), (2, 3)]  # vertex vi
# vj = [(4, 5), (6, 7)]  # vertex vj
#
# # Create an instance of the EdgeFeatures class
# edge_feature_calculator = EdgeFeatures()
#
# # Calculate edge features
# features = edge_feature_calculator.compute_edge_features(vi, vj)
# print("Edge features:", features)
