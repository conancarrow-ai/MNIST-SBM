"""Z1 grid graph construction and coloring utilities."""

import numpy as np
import networkx as nx

from thrml.block_management import Block
from thrml.pgm import SpinNode


class Z1Graph:
    """60x60 grid graph with extra knight's-move-style connections."""

    def __init__(self, grid_size=60, data_size=784, connections=((1,0),(2,1),(2,3),(4,1))):
        self.grid_size = grid_size
        self.graph = nx.grid_graph(dim=(grid_size, grid_size), periodic=False)

        self.coord_to_node = {coord: SpinNode() for coord in self.graph.nodes}
        nx.relabel_nodes(self.graph, self.coord_to_node, copy=False)
        for node, coord in ((v, k) for k, v in self.coord_to_node.items()):
            self.graph.nodes[node]["coords"] = coord

        self.coloring = nx.bipartite.color(self.graph)
        self.color_0 = [n for n, c in self.coloring.items() if c == 0]
        self.color_1 = [n for n, c in self.coloring.items() if c == 1]
        self.n_colors = max(self.coloring.values()) + 1

        for c in connections:
            self._wire(c)

        self.node_list = list(self.graph.nodes)
        self.edge_list = list(self.graph.edges)

        self.data_node_indices = np.random.choice(len(self.color_0), data_size, replace=False)
        self.data_nodes = [self.color_0[x] for x in self.data_node_indices]

    def _wire(self, c):
        a, b = c
        for n in self.graph:
            x, y = self.graph.nodes[n]["coords"]
            for m in [(x+a, y+b), (x-b, y+a), (x-a, y-b), (x+b, y-a)]:
                if 0 <= m[0] < self.grid_size and 0 <= m[1] < self.grid_size:
                    self.graph.add_edge(n, self.coord_to_node[m])


def color_blocks(coloring, n_colors, nodes, exclude_nodes=None):
    exclude = set(exclude_nodes) if exclude_nodes else set()
    groups = [[] for _ in range(n_colors)]
    for node in nodes:
        if node not in exclude:
            groups[coloring[node]].append(node)
    return [Block(g) for g in groups if g]
