import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import deque
from utils import *
from typing import Optional, List
from copy import deepcopy
from extended_transformations.crop_grid import crop_grid_based
from extended_transformations.connect_grid import connect_grid_based
from extended_transformations.magnet_grid import magnet_grid_based
from extended_transformations.upscale_grid import upscale_grid_based
from extended_transformations.rotate_grid import rotate_grid_based
from extended_transformations.mirror_grid import mirror_grid_based
from extended_transformations.fill_grid import fill_grid_based
from extended_transformations.beam_grid import beam_grid_based
from extended_transformations.recolor_grid import recolor_grid_based
from extended_transformations.shift_grid import shift_grid_based
from extended_transformations.truncate_grid import truncate_grid_based
from extended_transformations.rotate_duplicate import rotate_duplicate_grid_based
from extended_transformations.arbitrary_duplicate_grid import arbitrary_duplicate_grid_based

def swap_with_zero(grid):
    color = next(cell for row in grid for cell in row if cell != 0)
    return [[0 if cell == color else color for cell in row] for row in grid]

def count_unique_colors_except_zero(grid):
    """Count the number of unique colors in the grid, excluding 0."""
    unique_colors = set()
    for row in grid:
        for color in row:
            if color != 0:
                unique_colors.add(color)
    return len(unique_colors)

def count_most_frequent_color_except_zero(grid):
    color_count = {}
    for row in grid:
        for color in row:
            if color != 0:
                color_count[color] = color_count.get(color, 0) + 1
    return color_count

class ARCGraph:
    colors = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
              "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
    img_dir = "images"
    insertion_transformation_ops = ["insert"]
    param_binding_ops = ["param_bind_neighbor_by_size", "param_bind_neighbor_by_color", "param_bind_node_by_shape",
                         "param_bind_node_by_size"]
    
    filter_ops = ["filter_by_color", "filter_by_size", "filter_by_degree",
                  "filter_by_neighbor_size", "filter_by_neighbor_color"]

    
    transformation_ops = {
        "nbccg": ["extract", "update_color", "move_node", "extend_node", "move_node_max", "fill_rectangle", "hollow_rectangle",
                  "add_border", "insert", "mirror", "flip", "rotate_node", "remove_node"],
        "nbvcg": ["extract","update_color", "move_node", "extend_node", "move_node_max", "remove_node"],
        "nbhcg": ["extract","update_color", "move_node", "extend_node", "move_node_max", "remove_node"],
        "ccgbr": ["extract","update_color", "remove_node"],
        "ccgbr2": ["extract","update_color", "remove_node"],
        "ccg": ["extract","update_color", "remove_node"],
        "mcccg": ["extract","move_node", "move_node_max", "rotate_node", "fill_rectangle", "add_border", "insert", "mirror",
                  "flip", "remove_node"],
       "na": ["extract",  'duplicate', 'upscale_grid', "crop", "fill", "magnet", "beam", "shift",
                                  "arbitrary_duplicate", "rotate_duplicate",
                                  "mirror_grid", 'rotate_grid', "connect", "recolor", "truncate",
                 "move_node", "move_node_max", "update_color", "extend_node",
                  "rotate_node", "add_border", "fill_rectangle", "hollow_rectangle", 
                   "mirror", "flip", "insert", "remove_node"],
        "lrg": ["extract", "update_color", "move_node", "extend_node", "move_node_max"]
        }

    dynamic_parameters = {"color", "direction", "point", "mirror_point", "mirror_direction", "mirror_axis"}

    def __init__(self, graph, name, image, abstraction=None):

        self.graph = graph
        self.image = image
        self.abstraction = abstraction
        if abstraction is None:
            self.name = name
        elif abstraction in name.split("_"):
            self.name = name
        else:
            self.name = name + "_" + abstraction
        if self.abstraction in image.multicolor_abstractions:
            self.is_multicolor = True
            self.most_common_color = 0
            self.least_common_color = 0
        else:
            self.is_multicolor = False
            self.most_common_color = image.most_common_color
            self.least_common_color = image.least_common_color

        self.width = max([node[1] for node in self.image.graph.nodes()]) + 1
        self.height = max([node[0] for node in self.image.graph.nodes()]) + 1
        self.task_id = name.split("_")[0]
        self.save_dir = self.img_dir + "/" + self.task_id

    # Add these methods inside the ARCGraph class in ARCGraph.py
    #Registry From Box coordinates Visual Primitive
    def register_node_from_box(self, r1: int, c1: int, r2: int, c2: int, node_id: str = "vlm_selected_node"):
        """
        Converts a VLM bounding box primitive into an active ARCGraph node
        by harvesting non-background pixels within the box boundaries.
        """
        grid = self.graph_to_grid()
        node_pixels = []
        node_colors = []
        bg_color = self.image.background_color

        # Scan the bounding box area
        for r in range(max(0, r1), min(self.height, r2 + 1)):
            for c in range(max(0, c1), min(self.width, c2 + 1)):
                if grid[r][c] != bg_color:
                    node_pixels.append((r, c))
                    node_colors.append(grid[r][c])

        if not node_pixels:
            raise ValueError(f"No foreground object pixels found inside box [{r1}, {c1}, {r2}, {c2}]")

        # Clear existing node with this ID if it exists, then allocate a new one
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            
        self.graph.add_node(
            node_id, 
            nodes=node_pixels, 
            color=node_colors if self.is_multicolor else node_colors[0], 
            size=len(node_pixels)
        )
        return node_id
    #Registry From Point coordinates Visual Primitive
    def register_node_from_point(self, r: int, c: int, node_id: str = "vlm_selected_node"):
        """
        Converts a VLM point primitive into an active ARCGraph node
        using flood-fill (connected components) starting from the clicked pixel.
        """
        grid = self.graph_to_grid()
        bg_color = self.image.background_color
        
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"Point [{r}, {c}] is out of grid boundaries.")
            
        target_color = grid[r][c]
        if target_color == bg_color:
            raise ValueError(f"Selected point [{r}, {c}] lands on background canvas.")

        # Classic Breadth-First Search (BFS) to capture the connected component
        queue = deque([(r, c)])
        visited = set([(r, c)])
        node_pixels = []
        
        while queue:
            curr_r, curr_c = queue.popleft()
            node_pixels.append((curr_r, curr_c))
            
            # Check 4-connectivity or 8-connectivity depending on your preference
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in visited and grid[nr][nc] == target_color:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)

        self.graph.add_node(
            node_id, 
            nodes=node_pixels, 
            color=[target_color]*len(node_pixels) if self.is_multicolor else target_color, 
            size=len(node_pixels)
        )
        return node_id

    # ------------------------------------- filters ------------------------------------------
    #  filters take the form of filter(node, params), return true if node satisfies filter

    def graph_to_grid(self):
        """
        Converts the graph to a grid representation.
        """
        height = self.image.height
        width = self.image.width
        grid = [[self.image.background_color for _ in range(width)] for _ in range(height)]
        for node, data in self.graph.nodes(data=True):
            color = data.get('color', self.image.background_color)
            sub_nodes = data.get('nodes', [node])
            if isinstance(color, list):
                for idx, sub_node in enumerate(sub_nodes):
                    y, x = sub_node
                    grid[y][x] = color[idx]
            else:
                for y, x in sub_nodes:
                    grid[y][x] = color
        return grid

    def update_graph_from_grid(self, grid):
        """
        Updates the graph based on the grid representation.
        """
        self.graph = nx.grid_2d_graph(len(grid), len(grid[0]))
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                self.graph.nodes[(y, x)]['color'] = [color]
        self.image.grid = grid

    def filter_by_color(self, node, color: int, exclude: bool = False):
        if color == "most":
            # Exclude background color
            color_sizes = {}
            for n, data in self.graph.nodes(data=True):
                c = data["color"]
                size = data["size"]
                if isinstance(c, list):
                    # Multicolor node
                    for color_i in c:
                        if color_i == self.image.background_color:
                            continue
                        color_sizes[color_i] = color_sizes.get(color_i, 0) + 1  # Increment by 1 for each pixel
                else:
                    # Single color node
                    if c == self.image.background_color:
                        continue
                    color_sizes[c] = color_sizes.get(c, 0) + size  # Use size directly

            if color_sizes:
                # Get color with maximum total size
                color = max(color_sizes, key=lambda k: color_sizes[k])
            else:
                color = self.image.background_color

        elif color == "least":
            color_sizes = {}
            for n, data in self.graph.nodes(data=True):
                c = data["color"]
                size = data["size"]
                if isinstance(c, list):
                    for color_i in c:
                        if color_i == self.image.background_color:
                            continue
                        color_sizes[color_i] = color_sizes.get(color_i, 0) + 1  # Increment by 1 for each pixel
                else:
                    if c == self.image.background_color:
                        continue
                    color_sizes[c] = color_sizes.get(c, 0) + size

            if color_sizes:
                color = min(color_sizes, key=lambda k: color_sizes[k])
            else:
                color = self.image.background_color

        if self.is_multicolor:
            if not exclude:
                return color in self.graph.nodes[node]["color"]
            else:
                return color not in self.graph.nodes[node]["color"]
        else:
            if not exclude:
                return self.graph.nodes[node]["color"] == color
            else:
                return self.graph.nodes[node]["color"] != color



    def filter_by_size(self, node, size, exclude: bool = False):
        if size == "max":
            size = self.get_attribute_max("size")
        elif size == "min":
            size = self.get_attribute_min("size")
        if size == "odd" and not exclude:
            return self.graph.nodes[node]["size"] % 2 != 0
        elif size == "odd" and exclude:
            return self.graph.nodes[node]["size"] % 2 == 0
        elif not exclude:
            return self.graph.nodes[node]["size"] == size
        elif exclude:
            return self.graph.nodes[node]["size"] != size

    def filter_by_degree(self, node, degree, exclude: bool = False):
        if not exclude:
            return self.graph.degree[node] == degree
        else:
            return self.graph.degree[node] != degree

    def filter_by_neighbor_size(self, node, size, exclude: bool = False):
        if size == "max":
            size = self.get_attribute_max("size")
        elif size == "min":
            size = self.get_attribute_min("size")

        for neighbor in self.graph.neighbors(node):
            if size == "odd" and not exclude:
                if self.graph.nodes[neighbor]["size"] % 2 != 0:
                    return True
            elif size == "odd" and exclude:
                if self.graph.nodes[neighbor]["size"] % 2 == 0:
                    return True
            elif not exclude:
                if self.graph.nodes[neighbor]["size"] == size:
                    return True
            elif exclude:
                if self.graph.nodes[neighbor]["size"] != size:
                    return True
        return False

    def filter_by_neighbor_color(self, node, color, exclude: bool = False):
        if color == "same":
            color = self.graph.nodes[node]["color"]
        elif color == "most":
            color = self.most_common_color
        elif color == "least":
            color = self.least_common_color

        for neighbor in self.graph.neighbors(node):
            if not exclude:
                if self.graph.nodes[neighbor]["color"] == color:
                    return True
            elif exclude:
                if self.graph.nodes[neighbor]["color"] != color:
                    return True
        return False

    def filter_by_neighbor_degree(self, node, degree, exclude: bool = False):
        for neighbor in self.graph.neighbors(node):
            if not exclude:
                if self.graph.degree[neighbor] == degree:
                    return True
            else:
                if self.graph.degree[neighbor] != degree:
                    return True
        return False

    def param_bind_neighbor_by_color(self, node, color, exclude: bool = False):
        for neighbor in self.graph.neighbors(node):
            if self.filter_by_color(neighbor, color, exclude):
                return neighbor
        return None

    def param_bind_neighbor_by_size(self, node, size, exclude: bool = False):
        for neighbor in self.graph.neighbors(node):
            if self.filter_by_size(neighbor, size, exclude):
                return neighbor
        return None

    def param_bind_node_by_size(self, node, size, exclude: bool = False):
        for n in self.graph.nodes():
            if self.filter_by_size(n, size, exclude):
                return n
        return None

    def param_bind_neighbor_by_degree(self, node, degree, exclude: bool = False):
        for neighbor in self.graph.neighbors(node):
            if self.filter_by_degree(neighbor, degree, exclude):
                return neighbor
        return None

    def param_bind_node_by_shape(self, node):
        target_shape = self.get_shape(node)
        for param_bind_node in self.graph.nodes:
            if param_bind_node != node:
                candidate_shape = self.get_shape(param_bind_node)
                if candidate_shape == target_shape:
                    return param_bind_node
        return None

    def magnet(self, magnet_type = "dynamic", shifting_direction="dynamic",
               color1=0, color2=0, 
               #color3=0, color4=0,
               grid_size=0
               ):
        grid = self.graph_to_grid()
        transformed_grid = magnet_grid_based(grid, magnet_type, shifting_direction, color1, color2, grid_size)
        self.update_graph_from_grid(transformed_grid)
        return self
        
    def update_color(self, node, color):
        if color == "most":
            color = self.most_common_color
        elif color == "least":
            color = self.least_common_color
        self.graph.nodes[node]["color"] = color
        return self
    
    def fill(self, object, color, color1):
        grid = self.graph_to_grid()
        transformed_grid = fill_grid_based(grid, object, color, color1)
        self.update_graph_from_grid(transformed_grid)
        return self
    

    def connect(self, connect_mode: str, color: int, fill_color: int, border_color:int, inherit_vertical:bool):
        grid = self.graph_to_grid()
        transformed_grid = connect_grid_based(grid, connect_mode, color, fill_color, border_color, inherit_vertical)
        self.update_graph_from_grid(transformed_grid)
        return self
    
    def crop(self, corner: str = "right upper", crop_type: str = "corner_based", grid_size: int = 3,
            fill_color:int=0, border_color:int=0, fill_direction:str = "left_to_right", connect_all:bool=True):
        grid = self.graph_to_grid()
        transformed_grid = crop_grid_based(grid, corner, crop_type, grid_size, fill_color, border_color, fill_direction, connect_all)
        self.update_graph_from_grid(transformed_grid)
        return self        


    def extract(self, node, crop_filterless: bool=False, fraction: float=0.5):
        nodes_to_keep = node
        if nodes_to_keep is None:
            nodes_to_keep = list(self.graph.nodes())

        nodes_to_remove = [n for n in self.graph.nodes if n not in nodes_to_keep]
        self.graph.remove_nodes_from(nodes_to_remove)

        self.graph.remove_edges_from(list(self.graph.edges()))
        if not crop_filterless:
            return self
    
        all_sub_nodes = []
        for node in nodes_to_keep:
            data = self.graph.nodes[node]
            sub_nodes = data.get('nodes', [node])
            all_sub_nodes.extend(sub_nodes)

        if not all_sub_nodes:
            return self

        total_height = self.image.height
        total_width = self.image.width

        if total_height >= total_width:
            axis = 'vertical'
        else:
            axis = 'horizontal'

        ys = [n[0] for n in all_sub_nodes]
        xs = [n[1] for n in all_sub_nodes]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)

        if axis == 'horizontal':
            portion_width = max(1, int(total_width * fraction))
            new_min_x = 0
            new_max_x = portion_width - 1
            nodes_to_keep = [n for n in all_sub_nodes if new_min_x <= n[1] <= new_max_x]
        elif axis == 'vertical':
            portion_height = max(1, int(total_height * fraction))
            new_min_y = 0
            new_max_y = portion_height - 1
            nodes_to_keep = [n for n in all_sub_nodes if new_min_y <= n[0] <= new_max_y]
        else:
            return self

        self.graph.clear()
        for n in nodes_to_keep:
            self.graph.add_node(n)
            self.graph.nodes[n]['color'] = self.image.graph.nodes[n]['color']
            self.graph.nodes[n]['size'] = 1
            self.graph.nodes[n]['nodes'] = [n]
        return self



    def move_node(self, node, direction: Direction):
        assert direction is not None
        updated_sub_nodes = []
        delta_x = 0
        delta_y = 0
        if direction == Direction.UP or direction == Direction.UP_LEFT or direction == Direction.UP_RIGHT:
            delta_y = -1
        elif direction == Direction.DOWN or direction == Direction.DOWN_LEFT or direction == Direction.DOWN_RIGHT:
            delta_y = 1
        if direction == Direction.LEFT or direction == Direction.UP_LEFT or direction == Direction.DOWN_LEFT:
            delta_x = -1
        elif direction == Direction.RIGHT or direction == Direction.UP_RIGHT or direction == Direction.DOWN_RIGHT:
            delta_x = 1
        for sub_node in self.graph.nodes[node]["nodes"]:
            updated_sub_nodes.append((sub_node[0] + delta_y, sub_node[1] + delta_x))
        self.graph.nodes[node]["nodes"] = updated_sub_nodes
        return self

    def extend_node(self, node, direction: Direction, overlap: bool = False):
        assert direction is not None

        updated_sub_nodes = []
        delta_x = 0
        delta_y = 0
        if direction == Direction.UP or direction == Direction.UP_LEFT or direction == Direction.UP_RIGHT:
            delta_y = -1
        elif direction == Direction.DOWN or direction == Direction.DOWN_LEFT or direction == Direction.DOWN_RIGHT:
            delta_y = 1
        if direction == Direction.LEFT or direction == Direction.UP_LEFT or direction == Direction.DOWN_LEFT:
            delta_x = -1
        elif direction == Direction.RIGHT or direction == Direction.UP_RIGHT or direction == Direction.DOWN_RIGHT:
            delta_x = 1
        for sub_node in self.graph.nodes[node]["nodes"]:
            sub_node_y = sub_node[0]
            sub_node_x = sub_node[1]
            max_allowed = 1000
            for foo in range(max_allowed):
                updated_sub_nodes.append((sub_node_y, sub_node_x))
                sub_node_y += delta_y
                sub_node_x += delta_x
                if overlap and not self.check_inbound((sub_node_y, sub_node_x)):
                    break
                elif not overlap and (self.check_collision(node, [(sub_node_y, sub_node_x)])
                                      or not self.check_inbound((sub_node_y, sub_node_x))):
                    break
        self.graph.nodes[node]["nodes"] = list(set(updated_sub_nodes))
        self.graph.nodes[node]["size"] = len(updated_sub_nodes)

        return self

    def move_node_max(self, node, direction: Direction):
        assert direction is not None

        delta_x = 0
        delta_y = 0
        if direction == Direction.UP or direction == Direction.UP_LEFT or direction == Direction.UP_RIGHT:
            delta_y = -1
        elif direction == Direction.DOWN or direction == Direction.DOWN_LEFT or direction == Direction.DOWN_RIGHT:
            delta_y = 1
        if direction == Direction.LEFT or direction == Direction.UP_LEFT or direction == Direction.DOWN_LEFT:
            delta_x = -1
        elif direction == Direction.RIGHT or direction == Direction.UP_RIGHT or direction == Direction.DOWN_RIGHT:
            delta_x = 1
        max_allowed = 1000
        for foo in range(max_allowed):
            updated_nodes = []
            for sub_node in self.graph.nodes[node]["nodes"]:
                updated_nodes.append((sub_node[0] + delta_y, sub_node[1] + delta_x))
            if self.check_collision(node, updated_nodes) or not self.check_inbound(updated_nodes):
                break
            self.graph.nodes[node]["nodes"] = updated_nodes

        return self

    def rotate_node(self, node, rotation_dir: Rotation):
        rotate_times = 1
        if rotation_dir == Rotation.CW:
            mul = -1
        elif rotation_dir == Rotation.CCW:
            mul = 1
        elif rotation_dir == Rotation.CW2:
            rotate_times = 2
            mul = -1

        for t in range(rotate_times):
            center_point = (sum([n[0] for n in self.graph.nodes[node]["nodes"]]) // self.graph.nodes[node]["size"],
                            sum([n[1] for n in self.graph.nodes[node]["nodes"]]) // self.graph.nodes[node]["size"])
            new_nodes = []
            for sub_node in self.graph.nodes[node]["nodes"]:
                new_sub_node = (sub_node[0] - center_point[0], sub_node[1] - center_point[1])
                new_sub_node = (- new_sub_node[1] * mul, new_sub_node[0] * mul)
                new_sub_node = (new_sub_node[0] + center_point[0], new_sub_node[1] + center_point[1])
                new_nodes.append(new_sub_node)
            self.graph.nodes[node]["nodes"] = new_nodes
        return self

    def add_border(self, node, border_color):
        delta = [-1, 0, 1]
        border_pixels = []
        for sub_node in self.graph.nodes[node]["nodes"]:
            for x in delta:
                for y in delta:
                    border_pixel = (sub_node[0] + y, sub_node[1] + x)
                    if border_pixel not in border_pixels and not self.check_pixel_occupied(border_pixel):
                        border_pixels.append(border_pixel)
        new_node_id = self.generate_node_id(border_color)
        if self.is_multicolor:
            self.graph.add_node(new_node_id, nodes=list(border_pixels), color=[border_color for j in border_pixels],
                                size=len(border_pixels))
        else:
            self.graph.add_node(new_node_id, nodes=list(border_pixels), color=border_color, size=len(border_pixels))
        return self

    def fill_rectangle(self, node, fill_color, overlap: bool):
        if fill_color == "same":
            fill_color = self.graph.nodes[node]["color"]

        all_x = [sub_node[1] for sub_node in self.graph.nodes[node]["nodes"]]
        all_y = [sub_node[0] for sub_node in self.graph.nodes[node]["nodes"]]
        min_x, min_y, max_x, max_y = min(all_x), min(all_y), max(all_x), max(all_y)
        unfilled_pixels = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                pixel = (y, x)
                if pixel not in self.graph.nodes[node]["nodes"]:
                    if overlap:
                        unfilled_pixels.append(pixel)
                    elif not self.check_pixel_occupied(pixel):
                        unfilled_pixels.append(pixel)
        if len(unfilled_pixels) > 0:
            new_node_id = self.generate_node_id(fill_color)
            if self.is_multicolor:
                self.graph.add_node(new_node_id, nodes=list(unfilled_pixels),
                                    color=[fill_color for j in unfilled_pixels], size=len(unfilled_pixels))
            else:
                self.graph.add_node(new_node_id, nodes=list(unfilled_pixels), color=fill_color,
                                    size=len(unfilled_pixels))
        return self

    def hollow_rectangle(self, node, fill_color):
        all_y = [n[0] for n in self.graph.nodes[node]["nodes"]]
        all_x = [n[1] for n in self.graph.nodes[node]["nodes"]]
        border_y = [min(all_y), max(all_y)]
        border_x = [min(all_x), max(all_x)]
        non_border_pixels = []
        new_subnodes = []
        for subnode in self.graph.nodes[node]["nodes"]:
            if subnode[0] in border_y or subnode[1] in border_x:
                new_subnodes.append(subnode)
            else:
                non_border_pixels.append(subnode)
        self.graph.nodes[node]["nodes"] = new_subnodes
        if fill_color != self.image.background_color:
            new_node_id = self.generate_node_id(fill_color)
            self.graph.add_node(new_node_id, nodes=list(non_border_pixels), color=fill_color,
                                size=len(non_border_pixels))
        return self
    

    def truncate(self, color1, color2, grid_size, truncate_type, mirror):
        grid = self.graph_to_grid()
        transformed_grid = truncate_grid_based(grid, color1, color2, grid_size, truncate_type, mirror)
        self.update_graph_from_grid(transformed_grid)
        return self

    def shift(self, color1):
        grid = self.graph_to_grid()
        transformed_grid = shift_grid_based(grid, color1)
        self.update_graph_from_grid(transformed_grid)
        return self
    
    def recolor(self, recolor_type, color1, color2, shifting_direction):
        grid = self.graph_to_grid()
        transformed_grid = recolor_grid_based(grid, recolor_type, color1, color2, shifting_direction)
        self.update_graph_from_grid(transformed_grid)
        return self

    def upscale_grid(self, factor, mirror, upscale_type, color,
                     border_color, fill_color
                     ):
        grid = self.graph_to_grid()
        transformed_grid = upscale_grid_based(grid, factor, mirror, upscale_type, color,border_color, fill_color)
        self.update_graph_from_grid(transformed_grid)
        return self

    def rotate_grid(self, degrees):
        grid = self.graph_to_grid()
        transformed_grid = rotate_grid_based(grid, degrees)
        self.update_graph_from_grid(transformed_grid)
        return self


    def mirror_grid(self, mirror_axis="diagonal", mirror_type="color", color1:int=0, color2:int=0):
        grid = self.graph_to_grid()
        transformed_grid = mirror_grid_based(grid, mirror_axis, mirror_type, color1, color2)
        self.update_graph_from_grid(transformed_grid)
        return self


    def mirror(self, node, mirror_axis):
        if mirror_axis[1] is None and mirror_axis[0] is not None:
            axis = mirror_axis[0]
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = axis - (subnode[0] - axis)
                new_x = subnode[1]
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_axis[0] is None and mirror_axis[1] is not None:
            axis = mirror_axis[1]
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = subnode[0]
                new_x = axis - (subnode[1] - axis)
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        return self

    def flip(self, node, mirror_direction: Mirror):
        if mirror_direction == Mirror.VERTICAL:
            max_y = max([subnode[0] for subnode in self.graph.nodes[node]["nodes"]])
            min_y = min([subnode[0] for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = max_y - (subnode[0] - min_y)
                new_x = subnode[1]
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_direction == Mirror.HORIZONTAL:
            max_x = max([subnode[1] for subnode in self.graph.nodes[node]["nodes"]])
            min_x = min([subnode[1] for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_y = subnode[0]
                new_x = max_x - (subnode[1] - min_x)
                new_subnodes.append((new_y, new_x))
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_direction == Mirror.DIAGONAL_LEFT:  # \
            min_x = min([subnode[1] for subnode in self.graph.nodes[node]["nodes"]])
            min_y = min([subnode[0] for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_subnode = (subnode[0] - min_y, subnode[1] - min_x)
                new_subnode = (new_subnode[1], new_subnode[0])
                new_subnode = (new_subnode[0] + min_y, new_subnode[1] + min_x)
                new_subnodes.append(new_subnode)
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        elif mirror_direction == Mirror.DIAGONAL_RIGHT:  # /
            max_x = max([subnode[1] for subnode in self.graph.nodes[node]["nodes"]])
            min_y = min([subnode[0] for subnode in self.graph.nodes[node]["nodes"]])
            new_subnodes = []
            for subnode in self.graph.nodes[node]["nodes"]:
                new_subnode = (subnode[0] - min_y, subnode[1] - max_x)
                new_subnode = (- new_subnode[1], - new_subnode[0])
                new_subnode = (new_subnode[0] + min_y, new_subnode[1] + max_x)
                new_subnodes.append(new_subnode)
            if not self.check_collision(node, new_subnodes):
                self.graph.nodes[node]["nodes"] = new_subnodes
        return self

    def insert(self, node, object_id, point, relative_pos: RelativePosition):
        node_centroid = self.get_centroid(node)
        if not isinstance(point, tuple):
            if point == ImagePoints.TOP:
                point = (0, node_centroid[1])
            elif point == ImagePoints.BOTTOM:
                point = (self.image.height - 1, node_centroid[1])
            elif point == ImagePoints.LEFT:
                point = (node_centroid[0], 0)
            elif point == ImagePoints.RIGHT:
                point = (node_centroid[0], self.image.width - 1)
            elif point == ImagePoints.TOP_LEFT:
                point = (0, 0)
            elif point == ImagePoints.TOP_RIGHT:
                point = (0, self.image.width - 1)
            elif point == ImagePoints.BOTTOM_LEFT:
                point = (self.image.height - 1, 0)
            elif point == ImagePoints.BOTTOM_RIGHT:
                point = (self.image.height - 1, self.image.width - 1)
        if object_id == -1:
            object = self.graph.nodes[node]
        else:
            object = self.image.task.static_objects_for_insertion[self.abstraction][object_id]
        target_point = self.get_point_from_relative_pos(node_centroid, point, relative_pos)
        object_centroid = self.get_centroid_from_pixels(object["nodes"])
        subnodes_coords = []
        for subnode in object["nodes"]:
            delta_y = subnode[0] - object_centroid[0]
            delta_x = subnode[1] - object_centroid[1]
            subnodes_coords.append((target_point[0] + delta_y, target_point[1] + delta_x))
        new_node_id = self.generate_node_id(object["color"])
        self.graph.add_node(new_node_id, nodes=list(subnodes_coords), color=object["color"],
                            size=len(list(subnodes_coords)))
        return self

    def remove_node(self, node):
        self.graph.remove_node(node)

    # ------------------------------------- utils ------------------------------------------
    def get_attribute_max(self, attribute_name):
        if len(list(self.graph.nodes)) == 0:
            return None
        return max([data[attribute_name] for node, data in self.graph.nodes(data=True)])

    def get_attribute_min(self, attribute_name):
        if len(list(self.graph.nodes)) == 0:
            return None
        return min([data[attribute_name] for node, data in self.graph.nodes(data=True)])

    def get_color(self, node):
        if isinstance(node, list):
            return [self.graph.nodes[node_i]["color"] for node_i in node]
        else:
            return self.graph.nodes[node]["color"]

    def check_inbound(self, pixels):
        if not isinstance(pixels, list):
            pixels = [pixels]
        for pixel in pixels:
            y, x = pixel
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return False
        return True

    def check_collision(self, node_id, pixels_list=None):
        if pixels_list is None:
            pixels_set = set(self.graph.nodes[node_id]["nodes"])
        else:
            pixels_set = set(pixels_list)
        for node, data in self.graph.nodes(data=True):
            if len(set(data["nodes"]) & pixels_set) != 0 and node != node_id:
                return True
        return False

    def check_pixel_occupied(self, pixel):
        for node, data in self.graph.nodes(data=True):
            if pixel in data["nodes"]:
                return True
        return False

    def get_shape(self, node):
        sub_nodes = self.graph.nodes[node]["nodes"]
        if len(sub_nodes) == 0:
            return set()
        min_x = min([sub_node[1] for sub_node in sub_nodes])
        min_y = min([sub_node[0] for sub_node in sub_nodes])
        return set([(y - min_y, x - min_x) for y, x in sub_nodes])

    def get_centroid(self, node):
        """
        get the centroid of a node
        """
        center_y = (sum([n[0] for n in self.graph.nodes[node]["nodes"]]) + self.graph.nodes[node]["size"] // 2) // \
                   self.graph.nodes[node]["size"]
        center_x = (sum([n[1] for n in self.graph.nodes[node]["nodes"]]) + self.graph.nodes[node]["size"] // 2) // \
                   self.graph.nodes[node]["size"]
        return (center_y, center_x)

    def get_centroid_from_pixels(self, pixels):
        size = len(pixels)
        center_y = (sum([n[0] for n in pixels]) + size // 2) // size
        center_x = (sum([n[1] for n in pixels]) + size // 2) // size
        return (center_y, center_x)

    def get_relative_pos(self, node1, node2):
        for sub_node_1 in self.graph.nodes[node1]["nodes"]:
            for sub_node_2 in self.graph.nodes[node2]["nodes"]:
                if sub_node_1[0] == sub_node_2[0]:
                    if sub_node_1[1] < sub_node_2[1]:
                        return Direction.RIGHT
                    elif sub_node_1[1] > sub_node_2[1]:
                        return Direction.LEFT
                elif sub_node_1[1] == sub_node_2[1]:
                    if sub_node_1[0] < sub_node_2[0]:
                        return Direction.DOWN
                    elif sub_node_1[0] > sub_node_2[0]:
                        return Direction.UP
        return None

    def get_mirror_axis(self, node1, node2):
        node2_centroid = self.get_centroid(node2)
        if self.graph.edges[node1, node2]["direction"] == "vertical":
            return (node2_centroid[0], None)
        else:
            return (None, node2_centroid[1])

    def get_point_from_relative_pos(self, filtered_point, relative_point, relative_pos: RelativePosition):
        if relative_pos == RelativePosition.SOURCE:
            return filtered_point
        elif relative_pos == RelativePosition.TARGET:
            return relative_point
        elif relative_pos == RelativePosition.MIDDLE:
            y = (filtered_point[0] + relative_point[0]) // 2
            x = (filtered_point[1] + relative_point[1]) // 2
            return (y, x)

    # ------------------------------------------ apply -----------------------------------

    def apply(self, filters=None, filter_params=None, transformation=None, transformation_params=None):
        """
        Perform a full operation on the abstracted graph.
        """
        if not filters:
            for t, t_params in zip(transformation, transformation_params):
                getattr(self, t)(**t_params)
            return self
        if filters is None or not filters:
            self.apply_transformation(None, transformation, transformation_params[0])
            return self

        nodes_to_transform = []
        for node in self.graph.nodes():
            if self.apply_filters(node, filters, filter_params):
                nodes_to_transform.append(node)

        if not nodes_to_transform:
            return self

        params = self.apply_param_binding(nodes_to_transform, transformation_params)
        self.apply_transformation(nodes_to_transform, transformation, params)

    def apply_filters(self, node, filters, filter_params):
        """
        given filters and a node, return True if node satisfies all filters
        """
        satisfy = True
        for filter, filter_param in zip(filters, filter_params):
            satisfy = satisfy and getattr(self, filter)(node, **filter_param)
        return satisfy

    def apply_param_binding(self, node, transformation_params):
        """
        handle dynamic parameters: if a dictionary is passed as a parameter value, this means the parameter
        value needs to be retrieved from the parameter-binded nodes during the search

        example: set param "color" to the color of the neighbor with size 1
        """
        if isinstance(node, list):
            node = node[0]
        transformation_params_retrieved = copy.deepcopy(transformation_params[0])
        for param_key, param_value in transformation_params[0].items():
            if isinstance(param_value, dict):
                param_bind_function = param_value["filters"][0]
                param_bind_function_params = param_value["filter_params"][0]
                target_node = getattr(self, param_bind_function)(node, **param_bind_function_params)

                if param_key == "color":
                    target_color = self.get_color(target_node)
                    transformation_params_retrieved[param_key] = target_color
                elif param_key == "direction":
                    target_direction = self.get_relative_pos(node, target_node)
                    transformation_params_retrieved[param_key] = target_direction
                elif param_key == "mirror_point" or param_key == "point":
                    target_point = self.get_centroid(target_node)
                    transformation_params_retrieved[param_key] = target_point
                elif param_key == "mirror_axis":
                    target_axis = self.get_mirror_axis(node, target_node)
                    transformation_params_retrieved[param_key] = target_axis
                elif param_key == "mirror_direction":
                    target_mirror_dir = self.get_mirror_direction(node, target_node)
                    transformation_params_retrieved[param_key] = target_mirror_dir
                else:
                    raise ValueError("unsupported dynamic parameter")
        return transformation_params_retrieved

    def apply_transformation(self, nodes, transformation, transformation_params):
        """
        Apply transformation to a list of nodes or the entire graph.
        """
        if not isinstance(nodes, list):
            getattr(self, transformation[0])(nodes, **transformation_params) 
        elif nodes is None:
            # Apply transformations that operate on the entire graph
            getattr(self, transformation[0])(**transformation_params)
        elif transformation[0] == "extract":
            getattr(self, transformation[0])(nodes, **transformation_params)
        else:
            for node in nodes:
                getattr(self, transformation[0])(node, **transformation_params)

    # ------------------------------------------ meta utils -----------------------------------

    def copy(self):
        """
        return a copy of this ARCGraph object
        """
        return ARCGraph(self.graph.copy(), self.name, self.image, self.abstraction)

    def beam(self, color1=0, color2=0, beam_type:str="color_inheritance"):
        grid = self.graph_to_grid()
        transformed_grid = beam_grid_based(grid, color1, color2, beam_type)
        self.update_graph_from_grid(transformed_grid)
        return self
    
    def arbitrary_duplicate(self, mirror, duplicate_arbitrary, axis, mirror_grid, combine_pattern, concat_axis):
        grid = self.graph_to_grid()
        transformed_grid = arbitrary_duplicate_grid_based(grid, mirror, duplicate_arbitrary, axis, mirror_grid, combine_pattern, concat_axis)
        self.update_graph_from_grid(transformed_grid)
        return self
    
    def rotate_duplicate(self, mirror, rotation_degrees):
        grid = self.graph_to_grid()
        transformed_grid = rotate_duplicate_grid_based(grid, mirror, rotation_degrees)
        self.update_graph_from_grid(transformed_grid)
        return self

        

    def duplicate(self, axis: str ='horizontal', duplicate: int=2, color1:int=0,
                  mirror: bool=False,
                  #mirror_grid: Optional[str] = None,
                  concat_axis = "y",
                  #combine_pattern: str = "grid1 + grid2",
                  duplication_type: str="grid_based",
                  #rotation_degrees: List[int]=None
                  ):
        grid = self.graph_to_grid()
        if duplication_type == "pixel_based":
            def find_objects(grid, color1):
                visited = set()
                objects = []
                rows = len(grid)
                cols = len(grid[0]) if rows > 0 else 0

                directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]

                for i in range(rows):
                    for j in range(cols):
                        if (i, j) not in visited and grid[i][j] == color1:
                            queue = deque()
                            queue.append((i, j))
                            object_cells = set()

                            while queue:
                                x, y = queue.popleft()
                                if (x, y) in visited:
                                    continue
                                if grid[x][y] == color1:
                                    visited.add((x, y))
                                    object_cells.add((x, y))
                                    for dx, dy in directions:
                                        nx, ny = x + dx, y + dy
                                        if 0 <= nx < rows and 0 <= ny < cols:
                                            if (nx, ny) not in visited and grid[nx][ny] == color1:
                                                queue.append((nx, ny))
                            if object_cells:
                                objects.append(object_cells)
                return objects

            def find_replication_pixels(grid, obj_cells):
                replication_pixels = set()
                rows = len(grid)
                cols = len(grid[0]) if rows > 0 else 0

                for i in range(rows):
                    for j in range(cols):
                        if (i, j) not in obj_cells and grid[i][j] != 0:
                            replication_pixels.add((i, j, grid[i][j]))

                return list(replication_pixels)

            def crop_object(grid, obj_cells):

                if not obj_cells:
                    return []

                min_row = min(x for x, y in obj_cells)
                max_row = max(x for x, y in obj_cells)
                min_col = min(y for x, y in obj_cells)
                max_col = max(y for x, y in obj_cells)

                cropped = []
                for i in range(min_row, max_row + 1):
                    row = []
                    for j in range(min_col, max_col + 1):
                        val = grid[i][j] if (i, j) in obj_cells else 0
                        row.append(val)
                    cropped.append(row)
                return cropped

            def replicate_object(cropped_obj, color):

                replicated_obj = [[color if val != 0 else 0 for val in row] for row in cropped_obj]
                return replicated_obj
            objects = find_objects(grid, color1)
            desired_grid = []
            obj = objects[0]
            
            replication_pixels = find_replication_pixels(grid, obj)
            if not replication_pixels:
                raise ValueError("No Replication pixels!")
            rp_rows = [rp[0] for rp in replication_pixels]
            rp_cols = [rp[1] for rp in replication_pixels]

            if concat_axis=="xy":
                if all(r == rp_rows[0] for r in rp_rows):
                    arrangement = 'horizontal'
                    replication_pixels.sort(key=lambda x: x[1])
                elif all(c == rp_cols[0] for c in rp_cols):
                    arrangement = 'vertical'
                    replication_pixels.sort(key=lambda x: x[0])
                else:
                    raise ValueError('Replication pixels are not arranged strictly horizontally or vertically.')
            elif concat_axis == 'y':
                arrangement = "vertical"
            elif concat_axis == "x":
                arrangement = "horizontal"
                

            cropped_obj = crop_object(grid, obj)
            obj_height = len(cropped_obj)

            replicated_objects = []
            for rp in replication_pixels:
                _, _, rp_color = rp
                if mirror:
                    replicated_obj = replicate_object(cropped_obj, rp_color)
                else:
                    replicated_obj = replicate_object(cropped_obj, color1)
                replicated_objects.append(replicated_obj)

            if arrangement == 'horizontal':
                desired_grid = [[] for _ in range(obj_height)]
                for replica in replicated_objects:
                    for i in range(obj_height):
                        desired_grid[i].extend(replica[i])
            elif arrangement == 'vertical':
                desired_grid = []
                for replica in replicated_objects:
                    desired_grid.extend(replica)
            if not mirror and len(desired_grid) != duplicate:
                zero_row = [0] * len(desired_grid[0])
                desired_grid.insert(0, zero_row)
            self.update_graph_from_grid(desired_grid)
            return self
        
        if duplication_type == "sibling_pixel":
            grid = deepcopy(grid)
            
            rows = len(grid)
            cols = len(grid[0]) if rows > 0 else 0
            
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            components = []
            
            def bfs(start_r, start_c):
                q = deque()
                q.append((start_r, start_c))
                visited[start_r][start_c] = True
                component = [(start_r, start_c)]
                
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if grid[nx][ny] != 0 and not visited[nx][ny]:
                                visited[nx][ny] = True
                                q.append((nx, ny))
                                component.append((nx, ny))
                return component
            
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        comp = bfs(r, c)
                        components.append(comp)
            
            if len(components) != 2:
                raise ValueError("Grid does not contain exactly one object and one single pixel.")
            
            object_component = None
            single_pixel = None
            for comp in components:
                if len(comp) == 1:
                    single_pixel = comp[0]
                else:
                    object_component = comp
            
            if object_component is None or single_pixel is None:
                raise ValueError("Failed to identify object or single pixel.")
            
            single_r, single_c = single_pixel
            single_color = grid[single_r][single_c]
            
            matching_pixel = None
            for (r, c) in object_component:
                if grid[r][c] == single_color:
                    matching_pixel = (r, c)
                    break
            
            if matching_pixel is None:
                raise ValueError("No matching pixel found in the object.")
            
            obj_r, obj_c = matching_pixel
            relative_positions = []
            for (r, c) in object_component:
                dx = r - obj_r
                dy = c - obj_c
                relative_positions.append((dx, dy, grid[r][c]))
            
            single_new_r, single_new_c = single_pixel
            
            for (dx, dy, val) in relative_positions:
                new_r = single_new_r + dx
                new_c = single_new_c + dy
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    grid[new_r][new_c] = val
                else:
                    pass
            
            grid[single_new_r][single_new_c] = 0
            self.update_graph_from_grid(grid)
            return self
        
        if duplication_type == "grid_based":
            total_rows = len(grid)
            start_index = (total_rows // 2) + (total_rows % 2)
            bottom_half = grid[start_index:]
            mirrored_bottom = bottom_half[::-1]
            transformed_grid = mirrored_bottom + bottom_half
            self.update_graph_from_grid(transformed_grid)
            return self
                
        if duplication_type == "top_bottom_duplication":
            def mirror_grid(grid, axis=None):
                if axis == 'horizontal':
                    return grid[::-1]
                elif axis == 'vertical':
                    return [row[::-1] for row in grid]
                elif axis == 'both':
                    return [row[::-1] for row in grid[::-1]]
                else:
                    return grid
            def hconcat(grid1, grid2):
                return [row1 + row2 for row1, row2 in zip(grid1, grid2)]
            MhO = mirror_grid(grid, axis='horizontal')
            MvO = mirror_grid(grid, axis='vertical')
            MvMhO = mirror_grid(grid, axis='both')
            top = hconcat(MvMhO, MhO)
            bottom = hconcat(MvO, grid)
            transformed_grid = top + bottom
            self.update_graph_from_grid(transformed_grid)
            return self
    
        if duplication_type == "object_based":
            color_counts = {}
            for row in grid:
                for value in row:
                    color_counts[value] = color_counts.get(value, 0) + 1
            if mirror:
                most_common_color = max(color_counts, key=color_counts.get)
            else:
                most_common_color = min(color_counts, key=color_counts.get)
            horizontally_concatenated = [row + row for row in grid]
            upscale_factor = duplicate
            upscaled_grid = []
            for row in grid:
                upscaled_row = []
                for value in row:
                    upscaled_row.extend([value] * upscale_factor)
                for _ in range(upscale_factor):
                    upscaled_grid.append(upscaled_row.copy())
            positions_with_most_common_color = set()
            for i, row in enumerate(upscaled_grid):
                for j, value in enumerate(row):
                    if value == most_common_color:
                        positions_with_most_common_color.add((i, j))
            total_rows = len(upscaled_grid)
            total_cols = len(upscaled_grid[0]) if total_rows > 0 else 0
            all_positions = set((i, j) for i in range(total_rows) for j in range(total_cols))
            positions_to_zero = all_positions - positions_with_most_common_color
            extended_horizontally = []
            for i in range(len(horizontally_concatenated)):
                if i < len(grid):
                    extended_horizontally.append(horizontally_concatenated[i] + grid[i])
                else:
                    extended_horizontally.append(horizontally_concatenated[i] + [0] * len(grid[0]))
            vertically_concatenated_stage1 = extended_horizontally + extended_horizontally
            vertically_concatenated_final = vertically_concatenated_stage1 + extended_horizontally
            transformed_grid = []
            for i, row in enumerate(vertically_concatenated_final):
                new_row = []
                for j, value in enumerate(row):
                    if (i, j) in positions_to_zero:
                        new_row.append(0)
                    else:
                        new_row.append(value)
                transformed_grid.append(new_row)
            self.update_graph_from_grid(transformed_grid)
            return self
        if duplication_type == "unique_color":
            if mirror:
                unique_colors = count_unique_colors_except_zero(grid)
            else:
                unique_colors = len(grid)
        
            if concat_axis == "x":
                duplicated_grid = []
                for row in grid:
                    new_row = row * unique_colors
                    duplicated_grid.append(new_row)
            
            elif concat_axis == "y":
                duplicated_grid = grid * unique_colors
            
            elif concat_axis == "xy":
                duplicated_grid_horizontal = []
                for row in grid:
                    new_row = row * unique_colors
                    duplicated_grid_horizontal.append(new_row)
                duplicated_grid = duplicated_grid_horizontal * unique_colors
            
            else:
                raise ValueError("Invalid value for concat_axis. Use 'x', 'y', or 'xy'.")
    
            self.update_graph_from_grid(duplicated_grid)
            return self
        if duplicate == 2:
            grid = self.graph_to_grid()
            if mirror and axis == "vertical":
                vmirrored_grid = [row[::-1] for row in grid]
                transformed_grid = [original_row + mirrored_row for original_row, mirrored_row in zip(grid, vmirrored_grid)]
            
            elif mirror and axis == "horizontal":
                hmirrored_grid = grid[::-1]
                transformed_grid = grid + hmirrored_grid
            
            elif not mirror and axis == "vertical":
                transformed_grid = [row + row for row in grid]
            
            elif not mirror and axis == "horizontal":
                transformed_grid = grid + grid
            
            else:
                raise ValueError("Invalid combination of mirror and axis parameters.")
            
            self.update_graph_from_grid(transformed_grid)
            return self



        elif duplicate == 4:
            original_nodes = list(self.graph.nodes(data=True))
            original_width = self.width
            original_height = self.height
            self.width *= 2
            self.height *= 2
            transforms = [
                ('original', 0, 0),
                ('h_mirror', original_width, 0),
                ('v_mirror', 0, original_height),
                ('hv_mirror', original_width, original_height),
            ]

            max_ids = {}
            for node, data in original_nodes:
                node_colors = data['color']
                if isinstance(node_colors, list):
                    colors = node_colors
                else:
                    colors = [node_colors]
                for color in colors:
                    if color not in max_ids:
                        max_ids[color] = -1
                    if isinstance(node, tuple) and node[0] == color:
                        max_ids[color] = max(max_ids[color], node[1])

            for transform_name, shift_x, shift_y in transforms:
                if transform_name == 'original':
                    for node, data in original_nodes:
                        node_colors = data['color']
                        if isinstance(node_colors, list):
                            color = node_colors[0]
                        else:
                            color = node_colors
                        if color not in max_ids:
                            max_ids[color] = -1
                        max_ids[color] += 1
                        new_node_id = (color, max_ids[color])
                        subnodes = data.get('nodes', [node])
                        new_subnodes = [(y + shift_y, x + shift_x) for y, x in subnodes]
                        self.graph.add_node(new_node_id, nodes=new_subnodes, color=data['color'],
                                            size=data.get('size', len(new_subnodes)))
                elif transform_name == 'h_mirror':
                    for node, data in original_nodes:
                        node_colors = data['color']
                        if isinstance(node_colors, list):
                            color = node_colors[0]
                        else:
                            color = node_colors
                        if color not in max_ids:
                            max_ids[color] = -1
                        max_ids[color] += 1
                        new_node_id = (color, max_ids[color])
                        subnodes = data.get('nodes', [node])
                        new_subnodes = []
                        for y, x in subnodes:
                            mirrored_x = (original_width - 1) - x
                            new_subnodes.append((y, mirrored_x))
                        new_subnodes = [(y + shift_y, x + shift_x) for y, x in new_subnodes]
                        self.graph.add_node(new_node_id, nodes=new_subnodes, color=data['color'],
                                            size=data.get('size', len(new_subnodes)))
                elif transform_name == 'v_mirror':
                    for node, data in original_nodes:
                        node_colors = data['color']
                        if isinstance(node_colors, list):
                            color = node_colors[0]
                        else:
                            color = node_colors
                        if color not in max_ids:
                            max_ids[color] = -1
                        max_ids[color] += 1
                        new_node_id = (color, max_ids[color])
                        subnodes = data.get('nodes', [node])
                        new_subnodes = []
                        for y, x in subnodes:
                            mirrored_y = (original_height - 1) - y
                            new_subnodes.append((mirrored_y, x))
                        new_subnodes = [(y + shift_y, x + shift_x) for y, x in new_subnodes]
                        self.graph.add_node(new_node_id, nodes=new_subnodes, color=data['color'],
                                            size=data.get('size', len(new_subnodes)))
                elif transform_name == 'hv_mirror':
                    for node, data in original_nodes:
                        node_colors = data['color']
                        if isinstance(node_colors, list):
                            color = node_colors[0]
                        else:
                            color = node_colors
                        if color not in max_ids:
                            max_ids[color] = -1
                        max_ids[color] += 1
                        new_node_id = (color, max_ids[color])
                        subnodes = data.get('nodes', [node])
                        new_subnodes = []
                        for y, x in subnodes:
                            mirrored_y = (original_height - 1) - y  # Flip y
                            mirrored_x = (original_width - 1) - x  # Flip x
                            new_subnodes.append((mirrored_y, mirrored_x))
                        new_subnodes = [(y + shift_y, x + shift_x) for y, x in new_subnodes]
                        self.graph.add_node(new_node_id, nodes=new_subnodes, color=data['color'],
                                            size=data.get('size', len(new_subnodes)))
        else:
            raise ValueError(f"Unsupported duplicate value. Supported values are 2 and 4. You provided {duplicate}")

        return self


    def generate_node_id(self, color):
        """
        find the next available id for a given color,
        ex: if color=1 and there are already (1,0) and (1,1), return (1,2)
        """
        if isinstance(color, list):  # multi-color cases
            color = color[0]
        max_id = 0 
        for node in self.graph.nodes():
            if node[0] == color:
                max_id = max(max_id, node[1])
        return (color, max_id + 1)
    

    def undo_abstraction(self, adjust_to_bounding_box):
        if adjust_to_bounding_box:
            return self.undo_abstraction1()
        else:
            return self.undo_abstraction2()


    def undo_abstraction1(self):
        """
        Undo the abstraction to get the corresponding 2D grid.
        Return it as an ARCGraph object with adjusted size.
        """
        from image import Image
        nodes = []
        for component, data in self.graph.nodes(data=True):
            nodes_list = data.get('nodes', [component])
            nodes.extend(nodes_list)

        if not nodes:
            # If no nodes are present, return None
            return None

        # Determine the bounding box of the nodes
        ys = [node[0] for node in nodes]
        xs = [node[1] for node in nodes]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)

        # Calculate new width and height
        height = max_y - min_y + 1
        width = max_x - min_x + 1

        # Create an empty graph
        reconstructed_graph = nx.Graph()

        # Adjust node coordinates and set colors
        for component, data in self.graph.nodes(data=True):
            nodes_list = data.get('nodes', [component])
            color = data['color']
            if not isinstance(color, list):
                color = [color] * len(nodes_list)
            for node, c in zip(nodes_list, color):
                new_node = (node[0] - min_y, node[1] - min_x)
                reconstructed_graph.add_node(new_node)
                reconstructed_graph.nodes[new_node]["color"] = c

        # Create a new Image object with the adjusted size
        new_image = Image(
            self.image.task,
            width=width,
            height=height,
            graph=reconstructed_graph,
            name=self.image.name + "_reconstructed"
        )
        new_image.background_color = self.image.background_color

        return ARCGraph(reconstructed_graph, self.name + "_reconstructed", new_image, None)
    
    def undo_abstraction2(self):
        """
        undo the abstraction to get the corresponding 2D grid
        return it as an ARCGraph object
        """

        width, height = self.image.image_size
        reconstructed_graph = nx.grid_2d_graph(height, width)
        nx.set_node_attributes(reconstructed_graph, self.image.background_color, "color")

        if self.abstraction in self.image.multicolor_abstractions:
            for component, data in self.graph.nodes(data=True):
                for i, node in enumerate(data["nodes"]):
                    try:
                        reconstructed_graph.nodes[node]["color"] = data["color"][i]
                    except KeyError:
                        pass
        else:
            for component, data in self.graph.nodes(data=True):
                for node in data["nodes"]:
                    try:
                        reconstructed_graph.nodes[node]["color"] = data["color"]
                    except KeyError:
                        pass

        return ARCGraph(reconstructed_graph, self.name + "_reconstructed", self.image, None)


    def update_abstracted_graph(self, affected_nodes):
        """
        update the abstracted graphs so that they remain consistent after transformation
        """
        pixel_assignments = {}
        for node, data in self.graph.nodes(data=True):
            for subnode in data["nodes"]:
                if subnode in pixel_assignments:
                    pixel_assignments[subnode].append(node)
                else:
                    pixel_assignments[subnode] = [node]
        for pixel, nodes in pixel_assignments.items():
            if len(nodes) > 1:
                for node_1, node_2 in combinations(nodes, 2):
                    if not self.graph.has_edge(node_1, node_2):
                        self.graph.add_edge(node_1, node_2, direction="overlapping")

        for node1, node2 in combinations(self.graph.nodes, 2):
            if node1 == node2 or (
                    self.graph.has_edge(node1, node2) and self.graph.edges[node1, node2]["direction"] == "overlapping"):
                continue
            else:
                nodes_1 = self.graph.nodes[node1]["nodes"]
                nodes_2 = self.graph.nodes[node2]["nodes"]
                for n1 in nodes_1:
                    for n2 in nodes_2:
                        if n1[0] == n2[0]:  # two nodes on the same row
                            for column_index in range(min(n1[1], n2[1]) + 1, max(n1[1], n2[1])):
                                # try:
                                pixel_assignment = pixel_assignments.get((n1[0], column_index), [])
                                if len(pixel_assignment) == 0 or (len(pixel_assignment) == 1 and (
                                        pixel_assignment[0] == node1 or pixel_assignment[0] == node2)):
                                    continue
                                break
                            else:
                                if self.graph.has_edge(node1, node2):
                                    self.graph.edges[node1, node2]["direction"] = "horizontal"
                                else:
                                    self.graph.add_edge(node1, node2, direction="horizontal")
                                break
                        elif n1[1] == n2[1]:  # two nodes on the same column:
                            for row_index in range(min(n1[0], n2[0]) + 1, max(n1[0], n2[0])):
                                pixel_assignment = pixel_assignments.get((row_index, n1[1]), [])
                                if len(pixel_assignment) == 0 or (len(pixel_assignment) == 1 and (
                                        pixel_assignment[0] == node1 or pixel_assignment[0] == node2)):
                                    continue
                                break
                            else:
                                if self.graph.has_edge(node1, node2):
                                    self.graph.edges[node1, node2]["direction"] = "vertical"
                                else:
                                    self.graph.add_edge(node1, node2, direction="vertical")
                                break
                    else:
                        continue
                    break

    def plot(self, ax=None, save_fig=False, file_name=None):
        # """
        # Visualize the graph.
        # """
        # if ax is None:
        #     if self.abstraction is None:
        #         fig = plt.figure(figsize=(6, 6))
        #     else:
        #         fig = plt.figure(figsize=(4, 4))
        # else:
        #     fig = ax.get_figure()

        # if self.abstraction is None:
        #     pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}
        #     color = [self.colors[self.graph.nodes[x, y]["color"]] for x, y in self.graph.nodes()]

        #     nx.draw(self.graph, ax=ax, pos=pos, node_color=color, node_size=600)
        #     nx.draw_networkx_labels(self.graph, ax=ax, font_color="#676767", pos=pos, font_size=8)

        # else:
        #     pos = {}
        #     for node in self.graph.nodes:
        #         centroid = self.get_centroid(node)
        #         pos[node] = (centroid[1], -centroid[0])

        #     color = []
        #     for node, data in self.graph.nodes(data=True):
        #         if isinstance(data["color"], list):
        #             node_color = self.colors[data["color"][0]]
        #         else:
        #             node_color = self.colors[data["color"]]
        #         color.append(node_color)

        #     size = [300 * data["size"] for node, data in self.graph.nodes(data=True)]

        #     nx.draw(self.graph, pos=pos, node_color=color, node_size=size)
        #     nx.draw_networkx_labels(self.graph, font_color="#676767", pos=pos, font_size=8)

        #     edge_labels = nx.get_edge_attributes(self.graph, "direction")
        #     nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels)

        # if save_fig:
        #     if file_name is not None:
        #         fig.savefig(self.save_dir + "/" + file_name)
        #     else:
        #         fig.savefig(self.save_dir + "/" + self.name)
        # plt.close()
        pass
