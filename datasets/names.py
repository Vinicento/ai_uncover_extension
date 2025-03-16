import os

# Define the directory containing the images
directory = "natural_images"

# Get the list of files in the directory
files = os.listdir(directory)

# Sort files to maintain any original order
files.sort()

# Iterate through the files and rename them numerically starting from 0
for index, file in enumerate(files):
    # Extract the file extension
    extension = os.path.splitext(file)[1]
    # Create the new filename with the same extension
    new_name = f"natural_{index}{extension}"
    # Construct full file paths
    old_path = os.path.join(directory, file)
    new_path = os.path.join(directory, new_name)
    # Rename the file
    os.rename(old_path, new_path)

print("Files have been renamed numerically.")
