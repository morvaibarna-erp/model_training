import os
import shutil

def copy_common_files(folder1, folder2, output_folder):
    """
    Copies files with the same names from two source folders to an output folder.

    Parameters:
    folder1 (str): Path to the first folder.
    folder2 (str): Path to the second folder.
    output_folder (str): Path to the output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the set of filenames in both folders
    folder1_files = set(os.listdir(folder1))
    folder2_files = set(os.listdir(folder2))

    # Find the common files between the two folders
    common_files = folder1_files.intersection(folder2_files)

    # Copy each common file to the output folder
    for filename in common_files:
        file_path1 = os.path.join(folder1, filename)

        # Copy the file from folder1 (or folder2 if preferred)
        shutil.copy(file_path1, output_folder)
        print(f"Copied {filename} to {output_folder}")

# Example usage:
folder1 = './merok_kivalogatott'
folder2 = './red'
output_folder = './rednew'

copy_common_files(folder1, folder2, output_folder)
