import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from svgpathtools import Line, Arc
from svgpathtools import svg2paths

from data.compute_edge_features import EdgeFeatures
from data.compute_vertex_features import compute_vertex_features


# 解析svg文件
def parse_svg(svg_file):
    paths, attributes = svg2paths(svg_file)
    return paths, attributes


# 计算中点作为图神经网络的顶点
def mid_point(start, end):
    return (start[0] + end[0]) / 2, (start[1] + end[1]) / 2


# 计算两个点之间的欧几里得距离(可能存在像素和毫米之间的转换问题)
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 提取路径的中点
def extract_mid_points_and_segments(paths, attributes):
    """
    提取svg文件中的点，边的信息(
        line: 取线段起点和终点
        arc: 取弧线起点和终点的连接线段
        circle: 取圆的水平直径
        ellipse: 取椭圆的长轴
    )
    :param attributes: svg解析后的所有元素的属性字典
    :param paths: svg解析内容
    :return:
    points_info: [[(中点坐标), (起点坐标), (终点坐标)]]
    segment_types: ['line', 'arc', 'circle', 'ellipse'] 对应线的类型
    semantic_ids: [[semantic-id]]   对应点的类别
    instance_ids: [[instance-id]]
    """
    points_info = []
    segment_types = []
    semantic_ids = []

    for path, path_attributes in zip(paths, attributes):
        # 获取路径上的semantic-id
        semantic_id = path_attributes.get('semantic-id', None)
        for segment in path:
            # 添加semantic-id
            if semantic_id:
                semantic_ids.append([int(semantic_id)])
            else:
                semantic_ids.append([0])
            if isinstance(segment, Line):
                # 对于线段，取起点和终点的中点
                start_point = (segment.start.real, segment.start.imag)
                end_point = (segment.end.real, segment.end.imag)
                points_info.append([mid_point(start_point, end_point), start_point, end_point])
                segment_types.append('line')
            elif isinstance(segment, Arc):
                # 对于弧线，取起点和终点的中点
                start_point = (segment.start.real, segment.start.imag)
                end_point = (segment.end.real, segment.end.imag)
                points_info.append([mid_point(start_point, end_point), start_point, end_point])
                segment_types.append('arc')
            else:
                # 如果是其他类型的路径（如圆或椭圆）
                # 根据路径类型进行判断
                if hasattr(segment, 'center') and hasattr(segment, 'r1') and hasattr(segment, 'r2'):
                    # 这是一个椭圆
                    # 椭圆的长轴是其最大半轴，计算长轴两端的中点
                    major_axis_start = (segment.center.real - max(segment.r1, segment.r2), segment.center.imag)
                    major_axis_end = (segment.center.real + max(segment.r1, segment.r2), segment.center.imag)
                    points_info.append([mid_point(major_axis_start, major_axis_end), major_axis_start, major_axis_end])
                    segment_types.append('ellipse')

                elif hasattr(segment, 'center') and hasattr(segment, 'radius'):
                    # 这是一个圆
                    # 对于圆形，取水平直径
                    circle_start_point = (segment.center.real - segment.radius, segment.center.imag)
                    circle_end_point = (segment.center.real + segment.radius, segment.center.imag)
                    center = (segment.center.real, segment.center.imag)
                    points_info.append([center, circle_start_point, circle_end_point])
                    segment_types.append('ellipse')

    return points_info, segment_types, semantic_ids


# 构建图的函数，使用中点作为顶点
def build_graph_with_features(points_info, segment_types, semantic_ids, threshold_distance=300, max_edges_per_node=30):
    graph = nx.Graph()
    edge_feature_calculator = EdgeFeatures()

    # 获取中点信息
    mid_points = [point_info[0] for point_info in points_info]

    # 计算每个顶点的特征
    vertex_features = compute_vertex_features(mid_points, segment_types)

    # 为每个顶点添加节点和特征
    for i, (vertex, features) in enumerate(zip(mid_points, vertex_features)):
        graph.add_node(i, pos=vertex, features=features, target=semantic_ids[i][0])

    # 检查每对中点的距离，如果小于与之，才添加边
    for i in range(len(mid_points)):
        for j in range(i + 1, len(mid_points)):
            # 计算两个顶点之间的距离
            x1, y1 = mid_points[i]
            x2, y2 = mid_points[j]
            dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 如果距离小于阈值，且每个节点的边数不超过最大值，则添加边
            if dis < threshold_distance and len(list(graph.neighbors(i))) < max_edges_per_node and len(
                    list(graph.neighbors(j))) < max_edges_per_node:
                edge_feature = edge_feature_calculator.compute_edge_features(
                    [points_info[i][1], points_info[i][2]],
                    [points_info[j][1], points_info[j][2]]
                )
                graph.add_edge(i, j, features=edge_feature)

    return graph


# 可视化图
def draw_graph(graph):
    # 获取节点位置（如果你使用的是坐标）
    pos = nx.get_node_attributes(graph, 'pos')

    # 获取节点特征（例如，使用 feature 进行颜色或大小映射）
    features = nx.get_node_attributes(graph, 'feature')
    feature_values = np.array([f[2] for f in features.values()])  # 使用长度作为特征，或者其他特征

    # 创建绘图
    plt.figure(figsize=(8, 6))

    # 绘制节点，使用颜色映射
    node_size = 20  # 根据特征调整节点的大小
    node_color = feature_values  # 根据特征调整颜色

    # 绘制边
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, edge_color='black')

    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, cmap=plt.cm.Blues, alpha=0.7)

    # 绘制标签（如果需要）
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black', font_weight='bold')

    plt.title("Graph Visualization")
    plt.axis('off')  # 不显示坐标轴
    plt.show()
