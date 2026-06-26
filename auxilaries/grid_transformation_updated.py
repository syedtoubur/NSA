import networkx as nx
from image import Image
from ARCGraph import ARCGraph, Direction, Rotation, Mirror, ImagePoints, RelativePosition
import random
import shutil
import os
import numpy as np
from multiprocessing import Process, Queue
from PIL import Image as PILImage, ImageDraw, ImageFont
import json
from typing import Optional, List
from llm.selector_prompt import generate_selector_prompt
from plots import return_task_grid
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
import matplotlib.pyplot as plt

task_ids = ["3ee1011a"]

def are_grids_identical(grids):
    first_grid = grids[0]
    for grid in grids[1:]:
        if grid != first_grid:
            return False
    return True

def modify_grid(grid, trans_dict):
    height = len(grid)
    width = len(grid[0])
    background_color = 0
    image = Image(task=None, grid=grid, width=width, height=height)

    # Extract details from trans_dict
    abstraction = trans_dict['abstraction']
    filter_op = trans_dict['filter']
    filter_params = trans_dict['filter_params']
    transformation_op = trans_dict['transformation']
    transformation_params = trans_dict['transformation_params']
    parameter_binding = trans_dict.get('parameter_binding', None)  # Optional field

    # Define grid-based transformations
    grid_based_transformations = {
        "connect": connect_grid_based,
        "magnet": magnet_grid_based,
        "mirror_grid": mirror_grid_based,
        "upscale_grid": upscale_grid_based,
        "crop": crop_grid_based,
        "beam": beam_grid_based,
        "rotate_grid": rotate_grid_based,
        "recolor": recolor_grid_based,
        "shift": shift_grid_based,
        "truncate": truncate_grid_based,
        "rotate_duplicate": rotate_duplicate_grid_based,
        "fill": fill_grid_based,
        "arbitrary_duplicate": arbitrary_duplicate_grid_based
    }

    if transformation_op in grid_based_transformations:
        # Directly apply grid-based transformation
        try:
            modified_grid = grid_based_transformations[transformation_op](grid, **transformation_params)
            return modified_grid
        except Exception as e:
            print(f"Error applying grid-based transformation '{transformation_op}': {e}")
            return grid  # Return original grid if transformation fails

    elif transformation_op in ARCGraph.transformation_ops.get(abstraction, []):
        # Node-based transformations via ARCGraph
        abstracted_graph = getattr(image, Image.abstraction_ops[abstraction])()

        filtered_nodes = [
            node for node in abstracted_graph.graph.nodes()
            if abstracted_graph.apply_filters(node, [filter_op], [filter_params])
        ]

        if not filtered_nodes:
            print(f"No nodes matched for filter '{filter_op}' with parameters {filter_params}")
            return grid  # No transformation applied

        if transformation_op == 'extract':
            # Apply 'extract' once with all filtered nodes
            nodes_to_keep = filtered_nodes
            try:
                abstracted_graph.extract(nodes_to_keep, **transformation_params)
            except Exception as e:
                print(f"Error applying transformation '{transformation_op}' with nodes '{nodes_to_keep}': {e}")
        elif transformation_op == 'duplicate':
            # Apply 'duplicate' once with all parameters
            try:
                abstracted_graph.duplicate(**transformation_params)
            except Exception as e:
                print(f"Error applying transformation '{transformation_op}': {e}")
        else:
            # Apply transformation to each filtered node
            for node in filtered_nodes:
                # If parameter binding is specified, bind the parameters
                if parameter_binding:
                    param_bound_node = getattr(abstracted_graph, parameter_binding)(node, **filter_params)
                    if param_bound_node:
                        # Update transformation_params based on bound node if necessary
                        pass  # Implement as needed

                # Apply the transformation
                try:
                    getattr(abstracted_graph, transformation_op)(node, **transformation_params)
                except Exception as e:
                    print(f"Error applying transformation '{transformation_op}' on node '{node}': {e}")
                    continue  # Skip this node if transformation fails

        # Undo the abstraction to get the modified grid
        transformations_no_adjust = [
            "move_node", "move_node_max", "update_color", "extend_node",
            "rotate_node", "add_border", "fill_rectangle", "hollow_rectangle",
            "mirror", "flip", "insert", "remove_node", "extract",
        ]

        adjust_to_bounding_box = transformation_op not in transformations_no_adjust
        modified_image = abstracted_graph.undo_abstraction(adjust_to_bounding_box)
        if modified_image is None:
            print("Reconstruction Error: Modified image is None after undoing abstraction.")
            return grid  # Return original grid if reconstruction fails

        modified_grid = graph_to_grid(modified_image.graph, width, height, background_color)
        return modified_grid

    else:
        #print(f"Unsupported transformation operation: {transformation_op}")
        return grid  # Return original grid if transformation is unsupported

def graph_to_grid(graph, width, height, background_color):
    grid = [[background_color for _ in range(width)] for _ in range(height)]
    for node, data in graph.nodes(data=True):
        nodes = data.get("nodes", [node])
        color = data["color"]
        if not isinstance(color, list):
            color = [color] * len(nodes)
        for (y, x), c in zip(nodes, color):
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = c
    return grid

def save_grids_and_comparison(original_grid,
                              transformed_grid,
                              transformation_details,
                              path2save,
                              idx):
    def grid_to_image(grid):
        color_map = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
        color_map_rgb = [rgb_from_hex(c) for c in color_map]
        height, width = len(grid), len(grid[0])
        image = PILImage.new('RGB', (width, height))
        pixels = image.load()
        for y in range(height):
            for x in range(width):
                pixels[x, y] = color_map_rgb[grid[y][x]]
        return image

    original_image = grid_to_image(original_grid)
    transformed_image = grid_to_image(transformed_grid)
    space_between = 5
    comparison_image = PILImage.new('RGB', (original_image.width * 2 + space_between, original_image.height))
    comparison_image.paste(original_image, (0, 0))
    comparison_image.paste(transformed_image, (original_image.width + space_between, 0))

    # title_text = "\n".join(
    #     [f"Transformation {i + 1}: Abstraction: {detail['abstraction']}, Filter: {detail['filter']} {detail['filter_params']}, Transformation: {detail['transformation']} {detail['transformation_params']}"
    #      for i, detail in enumerate(transformation_details)]
    # )

    title_text = " ".join(transformation_details)

    # Save transformation details as a text file
    with open(f"{path2save}/transformation_info.txt", "w") as file:
        file.write(title_text)

    original_image.save(f"{path2save}/{idx}_original.png")
    transformed_image.save(f"{path2save}/{idx}_transformed.png")
    comparison_image.save(f"{path2save}/{idx}_comparison.png")

def prepare_and_save_transformed_data(grids, transformed_grids, transformation_details, output_folder="generated_llm_data"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    existing_files = [f for f in os.listdir(output_folder) if f.isdigit()]
    existing_numbers = [int(f) for f in existing_files if f.isdigit()]
    next_number = max(existing_numbers, default=0) + 1

    path2save = os.path.join(output_folder, str(next_number))
    os.makedirs(path2save)

    for index, (original_grid, transformed_grid) in enumerate(zip(grids, transformed_grids)):
        save_grids_and_comparison(original_grid,
                                  transformed_grid,
                                  transformation_details,
                                  path2save,
                                  index)
    save_transformation(grids, transformed_grids, transformation_details, path2save)

def save_transformation(grids, transformed_grids, transformation_details, path2save):
    data = {
        "instruction": "You are an advanced AI model specialized in solving Abstraction and Reasoning Corpus (ARC) tasks.",
        "input": generate_selector_prompt({"train": [{"input": grids[i], "output": transformed_grids[i]} for i in range(len(grids))]}),
        "output": transformation_details
    }
    output_path = os.path.join(path2save, "transformed_data.json")
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def append_transformation_to_file(file_path, grids, transformed_grids, transformation_details):
    new_transformation = {
        "instruction": "You are an advanced AI model specialized in solving Abstraction and Reasoning Corpus (ARC) tasks.",
        "input": generate_selector_prompt({"train": [{"input": grids[i], "output": transformed_grids[i]} for i in range(len(grids))]}),
        "output": " ".join(transformation_details)
    }

    if not os.path.exists(file_path):
        # Initialize file if it doesn't exist
        with open(file_path, 'w') as file:
            json.dump([], file)

    with open(file_path, 'r+') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = []
        data.append(new_transformation)
        file.seek(0)
        json.dump(data, file, indent=4)

def rgb_from_hex(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def load_images(sample_folder, samples):
    color_map = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                 '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']

    # Convert color_map from hex to RGB tuples
    color_map_rgb = [rgb_from_hex(c) for c in color_map]

    # List all image files in the folder
    image_files = [f for f in os.listdir(sample_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    if len(image_files) < samples:
        raise ValueError(f"Not enough images in {sample_folder} to sample {samples} images.")
    selected_files = random.sample(image_files, samples)  # Randomly select files

    # Load images and convert to grids
    grids = []
    for file in selected_files:
        image_path = os.path.join(sample_folder, file)
        with PILImage.open(image_path) as img:
            img = img.convert('RGB')  # Ensure image is in RGB mode
            img_array = np.array(img)

            # Initialize the grid with the same shape as img_array but only 2D
            grid = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=int)

            # Map each pixel to the closest color in color_map
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    pixel = tuple(img_array[i, j])
                    # Find the closest color index in color_map
                    distances = [np.linalg.norm(np.array(pixel) - np.array(color)) for color in color_map_rgb]
                    grid[i, j] = int(np.argmin(distances))

            grids.append(grid.tolist())

    return grids

def sample_and_apply_with_timeout(no_of_trans=1,
                                  sample_folder="full_video_data",
                                  samples=4,
                                  transformation_ops=None,
                                  chosen_task=None,
                                  timeout=2):
    """
    Wrapper function to apply sample_and_apply with a timeout.
    """
    def worker(q):
        try:
            result = sample_and_apply(
                no_of_trans=no_of_trans,
                sample_folder=sample_folder,
                samples=samples,
                transformation_ops=transformation_ops,
                chosen_task=chosen_task
            )
            q.put(result)
        except Exception as e:
            q.put(e)

    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None  # Indicate that the process timed out
    else:
        result = q.get()
        if isinstance(result, Exception):
            raise result
        return result



def sample_and_apply(no_of_trans=1,
                     sample_folder="full_video_data",
                     samples=4,
                     transformation_ops=None,
                     chosen_task=None):
    def grid_equal(grid1, grid2):
        if np.array(grid1).shape != np.array(grid2).shape:
            return False
        return all(grid1[i][j] == grid2[i][j] for i in range(len(grid1)) for j in range(len(grid1[0])))

    # Load samples
    if isinstance(samples, int):
        grids = load_images(sample_folder, samples)
    elif samples == "task_based":
        if chosen_task is None:
            chosen_task = np.random.choice(task_ids)
        grids_train = return_task_grid(chosen_task)["train"]
        grids = [grids_train[i]["input"] for i in range(len(grids_train))]

    start_grids = [grid[:] for grid in grids]  # Deep copy of the grids to preserve original state

    transformation_details = []
    all_trans_dicts =[]                    
    continue_prob = 0.3
    whether_to_continue = 1
    
    grid_height = len(grids[0])
    grid_width = len(grids[0][0])

    for idx in range(no_of_trans):
        if idx == 0:
            whether_to_continue = True
        else:
            if random.random() < continue_prob:
                whether_to_continue = True
                continue_prob = continue_prob ** 2
            else:
                whether_to_continue = False

        if idx != 0:
            if whether_to_continue:
                transformation_ops = None
            else:
                for _ in range(idx, no_of_trans):
                    transformation_details.append("no_trans")
                return start_grids, grids, transformation_details,all_trans_dicts
            



        # Force a hard choice between Node-based and Grid-based tracking first
        #track_type = random.choice(["node_based", "grid_based"]
        track_type="node_based" #Now for simplicity keeping only node based transformation
        changed = False
        no_of_skips = 0
        while not changed and no_of_skips<100:

            # Randomize Abstraction
            abstraction = random.choice(Image.abstractions)

            # Randomize Filter
            filter_op = random.choice(ARCGraph.filter_ops)
            filter_params = {}
            if filter_op == "filter_by_color":
                filter_params = {"color": random.randint(0, 9), "exclude": random.choice([True, False])}
            elif filter_op == "filter_by_size":
                filter_params = {"size": random.choice(["min", "max", "odd"]), "exclude": random.choice([True, False])}
            elif filter_op == "filter_by_degree":
                filter_params = {"degree": random.randint(1, 8), "exclude": random.choice([True, False])}
            elif filter_op == "filter_by_neighbor_size":
                filter_params = {"size": random.choice(["min", "max", "odd"]), "exclude": random.choice([True, False])}
            elif filter_op == "filter_by_neighbor_color":
                filter_params = {"color": random.randint(0, 9), "exclude": random.choice([True, False])}


            if track_type == "grid_based":
                # Only select from grid operations
                possible_trans_ops = ["connect", "magnet", "mirror_grid", "upscale_grid",
                                    "crop", "rotate_grid", "fill", "beam", "recolor", 
                                    "shift", "truncate", "arbitrary_duplicate", "rotate_duplicate"]
                transformation_op = random.choice(possible_trans_ops)
            else:
                # Only select from graph/node operations bound to this abstraction
                possible_trans_ops = ARCGraph.transformation_ops.get(abstraction, []) + ["extract", "duplicate"]
                if not possible_trans_ops:
                    no_of_skips += 1
                    continue
                transformation_op = random.choice(possible_trans_ops)


            transformation_params = {}
            if transformation_op == "update_color":
                transformation_params = {"color": random.randint(0, 9)}
                
            elif transformation_op in ["move_node", "extend_node", "move_node_max"]:
                transformation_params = {"direction": random.choice(list(Direction))}
                
            elif transformation_op == "rotate_node":
                transformation_params = {"rotation_dir": random.choice(list(Rotation))}
                
            elif transformation_op == "add_border":
                transformation_params = {"border_color": random.randint(0, 9)}
                
            elif transformation_op == "mirror_grid":
                transformation_params = {"mirror_axis": random.choice(["diagonal", "horizontal", "vertical"])}
                
            elif transformation_op == "connect":
                transformation_params = {
                    "connect_mode": random.choice([
                        "connect_rectangles", "connect_with_line", "connect_taxicab",
                        "connect_with_intersection", "connect_with_line", "cross_mode", "star_mode", "diagonal"
                    ]),
                    "color": random.randint(0, 9),
                    "fill_color": random.randint(0, 9),
                    "border_color": random.randint(0, 9),
                    "inherit_vertical": random.choice([True, False])
                }
            elif transformation_op == "magnet":
                transformation_params = {
                    "magnet_type": random.choice([
                                        "object",
                                        "magnet_line",
                                        "match_blank",
                                        'match_ver_union',
                                        "match_hor_union",
                                        "distract",
                                        "pixel",
                                        "corner_magnet",
                                        "whole_sort",
                                        "punch",
                                        "match_ver_line_union",
                                        "magnet_to_line",
                                        "match_hor_diff",
                                        "match_ver_no_line",
                                        "match_hor_no_line",
                                        "magnet_crop",
                    ]),
                    "shifting_direction": random.choice(["right", "left", "up", "down", "dynamic"]),
                    "color1": random.randint(0, 9),
                    "color2": random.randint(0, 9),
                    "grid_size": random.randint(1, 10)
                }
            elif transformation_op == "fill_rectangle":
                transformation_params = {"fill_color": random.randint(0, 9), "overlap": random.choice([True, False])}
            elif transformation_op == "hollow_rectangle":
                transformation_params = {"fill_color": random.randint(0, 9)}
            elif transformation_op == "mirror":
                axis_type = random.choice(['vertical', 'horizontal'])
                if axis_type == 'vertical':
                    axis_x = random.randint(0, grid_width - 1)
                    transformation_params = {"mirror_axis": (None, axis_x)}
                else:  # axis_type == 'horizontal'
                    axis_y = random.randint(0, grid_height - 1)
                    transformation_params = {"mirror_axis": (axis_y, None)}
                if axis_type == 'vertical':
                    axis_x = random.randint(0, grid_width - 1)  # grid[0] gives the width
                    transformation_params = {"mirror_axis": (None, axis_x)}
                else:  # axis_type == 'horizontal'
                    axis_y = random.randint(0, grid_height - 1)  # len(grid) gives the height
                    transformation_params = {"mirror_axis": (axis_y, None)}


            elif transformation_op == "flip":
                transformation_params = {"mirror_direction": random.choice(list(Mirror))}
            elif transformation_op == "insert":
                transformation_params = {
                    "object_id": -1,  # Assuming -1 means a default object
                    "point": random.choice(list(ImagePoints)),
                    "relative_pos": random.choice(list(RelativePosition))
                }
            elif transformation_op == "remove_node":
                transformation_params = {}
            elif transformation_op == "upscale_grid":
                transformation_params = {
                    "factor": random.randint(2, 4),  # Define the upscale factor (e.g., 2x, 3x, 4x)
                    "upscale_type": random.choice(["pixel_based",
                                "unique_colors",
                                "standard",
                                "other"]),
                    "mirror": random.choice([True, False]),
                    "color": random.randint(0, 9),
                    "border_color": random.randint(0, 9),
                    "fill_color": random.randint(0, 9)
                }
            elif transformation_op == "crop":
                transformation_params = {
                    "corner": random.choice(["left upper", "right upper", "left lower", "right lower"]),
                    "crop_type": random.choice([
                        "corner_based", "symetrics_based", "most_frequent_color_based_grid",
                        "most_frequent_color_based_flat", "delta_max", "delta_min",
                        "extract_colors", "extract_objects", "extract_colors_and_sort",
                        "nearest_corner_crop"
                    ]),
                    "grid_size": random.randint(2, 10),  # Define grid size, adjust as needed
                    "fill_color": random.randint(0, 9),
                    "border_color": random.randint(0, 9),
                    "fill_direction": random.choice(["left_to_right", "right_to_left", "up_to_down", "down_to_up"]),
                    "connect_all": random.choice([True, False])
                }
            elif transformation_op == "rotate_grid":
                transformation_params = {
                    "degrees": random.choice([0, 90, 180, 270])  # Define rotation angles
                }
            elif transformation_op == "fill":
                transformation_params = {
                    "object": random.choice([                    "empty_rectangle",
                    "empty_rectangle_dynamic",
                    "checkboard",
                    "maximal_square",
                    "fill_and_swap",]),
                    "color": random.randint(0, 9),
                     "color1": random.randint(0, 9)
                }
            elif transformation_op == "extract":
                transformation_params = {
                    "crop_filterless": random.choice([True, False]),
                    "fraction": random.uniform(0.1, 1.0)  # Random fraction between 0.1 and 1.0
                }
            elif transformation_op == "duplicate":
                duplication_types = ["top_bottom_duplication",
                                       "standard_duplication",
                                       "grid_based",
                                        "object_based",
                                        "pixel_based",
                                       "unique_color",
                                       "sibling_pixel",
                                       "rotation_based",]
                duplication_type = random.choice(duplication_types)
                transformation_params = {
                    "axis": random.choice(['horizontal', 'vertical']),
                    "duplicate": random.choice([2, 4]),
                    "mirror": random.choice([True, False]),
                    "concat_axis": random.choice(['x', 'y', "xy"]),
                    "duplication_type": duplication_type,
                }
            elif transformation_op == "beam":
                transformation_params = {
                    "color1": random.randint(0, 9),
                    "color2": random.randint(0, 9),
                    "beam_type": random.choice([
                        "box_based",
                        "color_inheritance",
                        "most_color_line",
                        "space_based",
                        "rectangle_shooting",
                        "linspace",
                        "infect",
                    ])
                }
            elif transformation_op == "recolor":  # Added handling for recolor_grid
                transformation_params = {
                    "recolor_type": random.choice([
                        "line_inheritance",
                        "border_based",
                        #"rule_based",
                        "nearest_pixels",
                        "fill_blank",
                        "square_spread",
                        "moving_recolor",
                    ]),
                    "color1": random.randint(0, 9),
                    "color2": random.randint(0, 9),
                    "shifting_direction": random.choice(["dynamic", "right", "left", "up", "down"])
                }
            elif transformation_op == "shift":  # Added handling for shift_grid
                transformation_params = {
                    "color1": random.randint(0, 9)
                }
            elif transformation_op == "truncate":  # Added handling for truncate_grid
                transformation_params = {
                    "color1": random.randint(0, 9),
                    "color2": random.randint(0, 9),
                    "grid_size": random.randint(1, 10),
                    "truncate_type": random.choice([
                        "position_based",
                        "inferior_based"
                    ]),
                    "mirror": random.choice([True, False])
                }
            elif transformation_op == "arbitrary_duplicate":  # Added handling for arbitrary_duplicate_grid
                transformation_params = {
                    "mirror": random.choice([True, False]),
                    "duplicate_arbitrary": random.randint(0, 5),
                    "axis": random.choice(['vertical', 'horizontal']),
                    "mirror_grid": random.choice([None, 'grid1', 'grid2']),
                    "combine_pattern": random.choice([
                                        "grid1+grid2",
                                        "grid1+grid2+grid1",
                                        "grid2+grid2+grid1",
                                        "grid1+grid1",
                                       "grid1 + grid1 + grid1 + grid2",
                                       "grid2 + grid1 + grid2 + grid1",
                                       "grid1+grid2+grid1+grid2+grid1"
                    ]),
                    "concat_axis": random.choice(['x', 'y', "xy"])
                }
            elif transformation_op == "rotate_duplicate":  # Added handling for rotate_duplicate_grid
                import itertools
                rotation_options = [0, 90, 180, 270]
                all_possible_values = list(itertools.product(rotation_options, repeat=4))
                transformation_params = {
                    "mirror": random.choice([True, False]),
                    "rotation_degrees": random.choice(all_possible_values)
                }
            else:
                # Handle other transformations or skip if unknown
                print(f"Unknown transformation: {transformation_op}. Skipping.")
                no_of_skips += 1
                continue

            trans_dict = {
                'abstraction': abstraction,
                'filter': filter_op,
                'filter_params': filter_params,
                'transformation': transformation_op,
                'transformation_params': transformation_params
            }

        # Execute and verify
            new_grids = []
            changed_flags = []
            for grid in grids:
                original_grid = [row[:] for row in grid]
                transformed_grid = modify_grid(grid, trans_dict)
                changed_flags.append(not grid_equal(original_grid, transformed_grid))
                new_grids.append(transformed_grid)

            if not are_grids_identical(new_grids) and all(changed_flags):
                changed = True
                grids = new_grids
                transformation_details.append(trans_dict["transformation"])
                all_trans_dicts.append(trans_dict)
            else:
                no_of_skips += 1
                # CRITICAL: If node-based failed, do NOT switch to grid_based. 
                # Stay on node_based and let the loop try a different filter/op combo.



    if changed:
        return start_grids, grids, transformation_details,all_trans_dicts
    else:
        raise Exception("No change")
