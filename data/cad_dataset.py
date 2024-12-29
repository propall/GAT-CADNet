from pathlib import Path

from torch_geometric.data import Dataset

from data.graph_to_tg_data import convert_nx_to_tg_data
from data.svg_to_graph import parse_svg, extract_mid_points_and_segments, build_graph_with_features


class CADDataset(Dataset):
    def __init__(self, svg_path):
        super().__init__()
        svg_paths = [str(f) for f in Path(svg_path).glob('*.svg')]
        self.svg_paths = svg_paths

    def __len__(self):
        return len(self.svg_paths)

    def __getitem__(self, item):
        file_path = self.svg_paths[item]

        # 获取对应的graph
        paths, attributes = parse_svg(file_path)
        points_info, segment_types, semantic_ids = extract_mid_points_and_segments(paths, attributes)
        graph = build_graph_with_features(points_info, segment_types, semantic_ids)

        data = convert_nx_to_tg_data(graph)

        return data


