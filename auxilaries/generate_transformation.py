import argparse
import os
import shutil
import json
from auxilaries.grid_transformation import sample_and_apply_with_timeout, prepare_and_save_transformed_data, append_transformation_to_file
from plots import return_task_grid
from tqdm import tqdm

def clear_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def initialize_json_file(json_path):
    with open(json_path, 'w') as f:
        json.dump([], f)

def update_progress_bar(folder, pbar, target_count):
    current_count = len(os.listdir(folder))
    pbar.n = current_count
    pbar.last_print_n = current_count
    pbar.refresh()

from tqdm import tqdm

def generate_samples(number_of_samples, output_folder, all_transformations_path, no_of_trans, transformation_ops=None, chosen_task=None):
    pbar = tqdm(total=number_of_samples, desc=f"Generating {no_of_trans} transformation samples")
    
    for _ in range(number_of_samples):
        try:
            # Use the modified function with timeout
            result = sample_and_apply_with_timeout(
                no_of_trans=no_of_trans,
                transformation_ops=transformation_ops,
                samples="task_based",
                chosen_task=chosen_task,
                timeout=0.1  # Timeout in seconds. If you cannot produce task in 0.1 sec then switch to next one.
            )
            if result is None:
                print("Sample generation timed out. Moving to the next transformation.")
                continue  # Skip to the next iteration

            original_grids, transformed_grids, transformation_details,all_trans_dicts = result
            prepare_and_save_transformed_data(original_grids, transformed_grids, all_trans_dicts, output_folder=output_folder)
            append_transformation_to_file(all_transformations_path, original_grids, transformed_grids, all_trans_dicts)
            update_progress_bar(output_folder, pbar, number_of_samples)
            if pbar.n >= number_of_samples:
                break
        except Exception as e:
            if str(e) == "No change":
                continue
            print(e)
    
    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Generate transformation samples and save them to specified directories.")
    parser.add_argument('--samples', type=int, default=500000, help='Number of samples to generate for each transformation type.')
    parser.add_argument('--one_trans_folder', type=str, default="final_data8", help='Output folder for one transformation samples.')
    parser.add_argument('--two_trans_folder', type=str, default="generated_llm_data_two_trans1", help='Output folder for two transformation samples.')
    parser.add_argument('--all_transformations_path', type=str, default="full_trans.json", help='Path to the JSON file storing all transformations.')
    parser.add_argument('--transformations', type=str, choices=['one', 'two', 'both'], default='one', help='Specify which transformations to perform: "one", "two", or "both".')
    parser.add_argument('--transformation_op', type=str, nargs='*', help='Specify the transformation operators to be used in sample_and_apply.')

    args = parser.parse_args()

    # Clear existing directories and create new ones based on the selected transformation
    if args.transformations in ['one', 'both']:
        print(f"DELETING: {args.one_trans_folder}")
        clear_and_create_folder(args.one_trans_folder)
    if args.transformations in ['two', 'both']:
        print(f"DELETING: {args.two_trans_folder}")
        clear_and_create_folder(args.two_trans_folder)

    # Initialize or clear the all_transformations.json file
    initialize_json_file(args.all_transformations_path)

    # Parse the transformation operators
    transformation_ops = args.transformation_op if args.transformation_op else None
    
    generate_samples(args.samples, args.one_trans_folder, args.all_transformations_path,
                     no_of_trans=3, transformation_ops=args.transformation_op)

if __name__ == "__main__":
    main()
