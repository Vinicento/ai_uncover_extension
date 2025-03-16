import os
from PIL import Image
from tqdm import tqdm

def convert_images_to_jpg(input_folder):
    supported_formats = ('.jpeg', '.png', '.jpg', '.bmp', '.gif', '.tiff')

    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(supported_formats):
            file_path = os.path.join(input_folder, filename)
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGBA')  # Convert to RGBA to handle transparency
                    img = img.convert('RGB')  # Convert to RGB to drop alpha channel

                    # Save as .jpg with the same filename but .jpg extension
                    output_file_path = os.path.join(input_folder, f"{os.path.splitext(filename)[0]}.jpg")
                    img.save(output_file_path, 'JPEG', quality=100)

                    # Remove the original file if it had a different extension
                    if file_path != output_file_path:
                        os.remove(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    input_folder = 'natural_images'  # Replace with your input folder path
    convert_images_to_jpg(input_folder)
