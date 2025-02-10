"""
Source: GAT-CADNet Paper (Pg3, equation 2)
Used to calculate the features of each vertex in the graph
vertex_feature = [cos(2a_i), sin(2a_i), l_i, t_i]
where a is the angle between the line from the coordinate origin to the vertex and the x-axis
l_i: Length of vᵢ sqrt(x² + y²)
t_i: One-hot encoding for arcs, segments, circles and ellipses

"""

import numpy as np


# Calculate angle and length
def compute_angle_and_length(x, y):
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += np.pi  # Ensure the angle is between [0, pi)
    # Calculate length
    length = np.sqrt(x**2 + y**2)
    return angle, length


# Calculate direction characteristics
def compute_direction_features(angle):
    return np.cos(2 * angle), np.sin(2 * angle)


# Create one-hot encoding (type)
def one_hot_coding(segment_type):
    types = ['line', 'arc', 'circle', 'ellipse']
    one_hot = np.zeros(4)
    one_hot[types.index(segment_type)] = 1
    return one_hot


# Calculate vertex features
def compute_vertex_features(vertices, segment_types):
    features = []
    for (x, y), segment_type in zip(vertices, segment_types):
        # Calculate angle and length
        angle, length = compute_angle_and_length(x, y)
        # Calculate angle and length
        cos_angle, sin_angle = compute_direction_features(angle)
        # Create one-hot encoding
        type_encoding = one_hot_coding(segment_type)
        # Combine features into a vector
        feature = [cos_angle, sin_angle, length] + type_encoding.tolist()
        features.append(feature)
    return np.array(features)
