from pathlib import Path

from torch_geometric.data import Dataset, Data, Batch

from data.graph_to_tg_data import convert_nx_to_tg_data
from data.svg_to_graph import parse_svg, extract_mid_points_and_segments, build_graph_with_features


class CADDataset(Dataset):
    """
    Standard Dataset class to process individual SVGs into graphs. (the original code from https://github.com/Liberation-happy/GAT-CADNet)
    Args:
        svg_path (str) : Directory path that contains svg images
        
    
    """
    def __init__(self, svg_path):
        super().__init__()
        # Find all SVG files in the directory
        svg_paths = [str(f) for f in Path(svg_path).glob('*.svg')]
        self.svg_paths = svg_paths # List of svg files

    def __len__(self):
        return len(self.svg_paths)

    def __getitem__(self, item):
        file_path = self.svg_paths[item]

        # Get the corresponding graph
        paths, attributes = parse_svg(file_path) # Detailed explanation in svg_to_graph.py

        points_info, segment_types, semantic_ids = extract_mid_points_and_segments(paths, attributes)
        graph = build_graph_with_features(points_info, segment_types, semantic_ids)

        # Convert networkx graph to PyG Data format
        """        
        |Attribute	     | Shape	                     | Description
        |data.x	         | (num_nodes, feature_dim)	     | Node feature matrix
        |data.edge_index | (2, num_edges)	             | Edge connectivity matrix
        |data.edge_attr	 | (num_edges, edge_feature_dim) | Edge feature matrix
        |data.y	         | (num_nodes,)	                 | Target labels for node classification
        |data.adj_matrix | (num_nodes, num_nodes)	     | Adjacency matrix
        """
        data = convert_nx_to_tg_data(graph)
        return data


