import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from svgpathtools import Line, Arc
from svgpathtools import svg2paths

from data.compute_edge_features import EdgeFeatures
from data.compute_vertex_features import compute_vertex_features

"""
Assume the svg file has an equilateral triangle, its svg file looks like this:

<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <path d="M 50 150 L 150 150 L 100 50 Z" stroke="black" stroke-width="2" fill="none"/>
</svg>

Command 	| Meaning	            | Coordinates
M 50 150	| Move To (50, 150)	    | Starts the path at (50, 150)
L 150 150	| Line To (150, 150)	| Draws a straight line to (150, 150)
L 100 50	| Line To (100, 50)	    | Draws a straight line to (100, 50)
Z	        | Close Path	        | Connects back to the starting point (50, 150), closing the shape

(1) paths, attributes = parse_svg("triangle.svg")

# Output:
paths = [
    [(50, 150), (150, 150)],
    [(150, 150), (100, 50)],
    [(100, 50), (50, 150)]
]

attributes = [
    {"stroke": "black", "stroke-width": 2, "fill": "none"},
    {"stroke": "black", "stroke-width": 2, "fill": "none"},
    {"stroke": "black", "stroke-width": 2, "fill": "none"}
]

"""

# Parse svg files
def parse_svg(svg_file):
    paths, attributes = svg2paths(svg_file)
    return paths, attributes


# Compute midpoints as vertices of graph neural networks
def mid_point(start, end):
    return (start[0] + end[0]) / 2, (start[1] + end[1]) / 2


# Calculate the Euclidean distance between two points (there may be conversion issues between pixels and millimeters)
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Extract the midpoint of a path
def extract_mid_points_and_segments(paths, attributes):
    """
    Extracts key points and segment types from SVG path data
    Args:
        paths (list) : List of path commands describing the geometric shape(drawing outlines in the image).
        attributes (list) : List of properties associated with each path
            Keys: "semantic-id", 
    Returns:
        points_info: Midpoints and endpoints of each segment. [[(midpoint coordinates), (starting point coordinates), (end point coordinates)]]
        segment_types: Stores the type of geometric segment ('line', 'arc', 'circle', 'ellipse')
        semantic_ids: Stores ids if present, else stores None

    
    Extract point and edge information from svg files(
        line: Get the starting point and end point of the line segment
        arc: Get the connecting line segment between the starting point and the end point of the arc
        circle: Take the horizontal diameter of the circle
        ellipse: Take the major axis of the ellipse
    )
    :param attributes: Attribute dictionary of all elements after svg parsing
    :param paths: svg parsing content
    :return:

    instance_ids: [[instance-id]]
    """
    points_info = [] # Stores midpoints and endpoints of each segment.
    segment_types = [] # Stores the type of geometric segment ('line', 'arc', 'circle', 'ellipse')
    semantic_ids = []

    # Using ".get()" instead of path_attributes['semantic-id'] avoids errors if the key doesn’t exist and allows setting a default fallback value(None)
    for path, path_attributes in zip(paths, attributes):
        # Get the semantic-id on the path
        semantic_id = path_attributes.get('semantic-id', None) # Look for the key 'semantic-id' and if not found return None 
        for segment in path:
            # add semantic-id
            if semantic_id:
                semantic_ids.append([int(semantic_id)])
            else:
                semantic_ids.append([0])
            if isinstance(segment, Line):
                # For line segments, take the midpoint of the start and end points
                start_point = (segment.start.real, segment.start.imag)
                end_point = (segment.end.real, segment.end.imag)
                points_info.append([mid_point(start_point, end_point), start_point, end_point])
                segment_types.append('line')
            elif isinstance(segment, Arc):
                # For arcs, take the midpoint of the start and end points
                start_point = (segment.start.real, segment.start.imag)
                end_point = (segment.end.real, segment.end.imag)
                points_info.append([mid_point(start_point, end_point), start_point, end_point])
                segment_types.append('arc')
            else:
                # If it is another type of path (such as a circle or ellipse)
                # Determine based on path type
                if hasattr(segment, 'center') and hasattr(segment, 'r1') and hasattr(segment, 'r2'):
                    # This is an ellipse
                    # The major axis of the ellipse is its largest semi-axis. Calculate the midpoints of the two ends of the major axis.
                    major_axis_start = (segment.center.real - max(segment.r1, segment.r2), segment.center.imag)
                    major_axis_end = (segment.center.real + max(segment.r1, segment.r2), segment.center.imag)
                    points_info.append([mid_point(major_axis_start, major_axis_end), major_axis_start, major_axis_end])
                    segment_types.append('ellipse')

                elif hasattr(segment, 'center') and hasattr(segment, 'radius'):
                    # This is a circle
                    # For circles, take the horizontal diameter
                    circle_start_point = (segment.center.real - segment.radius, segment.center.imag)
                    circle_end_point = (segment.center.real + segment.radius, segment.center.imag)
                    center = (segment.center.real, segment.center.imag)
                    points_info.append([center, circle_start_point, circle_end_point])
                    segment_types.append('ellipse')

    return points_info, segment_types, semantic_ids


# Functions for building graphs, using midpoints as vertices
def build_graph_with_features(points_info, segment_types, semantic_ids, threshold_distance=300, max_edges_per_node=30):
    """
    Constructs graph using midpoints of SVG path segments, connects nodes based on their distances, and assigns features to nodes and edges.
    threshold_distance is epsilon in paper Pg 3 while max_edges_per_node is K in paper Pg 3(above Instance and subgraph section)
    """
    
    
    graph = nx.Graph() # Create an empty graph
    edge_feature_calculator = EdgeFeatures() # A class that computes edge features.

    # Get midpoint information
    mid_points = [point_info[0] for point_info in points_info]

    # Calculate the characteristics of each vertex(using the midpoints for segment related calculation)
    vertex_features = compute_vertex_features(mid_points, segment_types)

    # Add nodes and features for each vertex
    for i, (vertex, features) in enumerate(zip(mid_points, vertex_features)):
        graph.add_node(i, pos=vertex, features=features, target=semantic_ids[i][0])

    # Check the distance of each pair of midpoints, and if it is less than, add edges
    for i in range(len(mid_points)):
        for j in range(i + 1, len(mid_points)):
            # Calculate the distance between two vertices
            x1, y1 = mid_points[i]
            x2, y2 = mid_points[j]
            dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # If the distance is less than the threshold and the number of edges of each node does not exceed the maximum value, add an edge
            if dis < threshold_distance and len(list(graph.neighbors(i))) < max_edges_per_node and len(
                    list(graph.neighbors(j))) < max_edges_per_node:
                edge_feature = edge_feature_calculator.compute_edge_features(
                    [points_info[i][1], points_info[i][2]],
                    [points_info[j][1], points_info[j][2]]
                )
                graph.add_edge(i, j, features=edge_feature)

    return graph


# Visualization
def draw_graph(graph):
    # Get the node location (if you are using coordinates)
    pos = nx.get_node_attributes(graph, 'pos')

    # Get node features (for example, color or size mapping using feature)
    features = nx.get_node_attributes(graph, 'feature')
    feature_values = np.array([f[2] for f in features.values()])  # Use length as a feature, or other features

    # Create a drawing
    plt.figure(figsize=(8, 6))

    # Draw nodes, using color maps
    node_size = 20  # Adjust the size of the node according to the characteristics
    node_color = feature_values  # Adjust color according to features

    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, edge_color='black')

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, cmap=plt.cm.Blues, alpha=0.7)

    # Draw the label (if needed)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black', font_weight='bold')

    plt.title("Graph Visualization")
    plt.axis('off')  # The coordinate axes are not displayed
    plt.show()
