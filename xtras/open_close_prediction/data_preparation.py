import os
import shutil
from PIL import Image

def combine_folders(folder1, folder2, folder3, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    index = 1

    def copy_and_convert_images(src_folder, index):
        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            if os.path.isfile(src_file):
                dest_file = os.path.join(destination_folder, f"image_{index}.jpg")
                try:
                    # Open the image file
                    with Image.open(src_file) as img:
                        # Convert to RGB and save as JPG
                        img.convert("RGB").save(dest_file, "JPEG")
                except Exception as e:
                    print(f"Error converting {src_file}: {e}")
                index += 1
        return index

    index = copy_and_convert_images(folder1, index)
    index = copy_and_convert_images(folder2, index)
    #copy_and_convert_images(folder3, index)

# Example usage
folder1 = 'Closed_Cabinets'
folder2 = ''
folder3 = ''
destination_folder = 'Cabinets_Closed'

combine_folders(folder1, folder2, folder3, destination_folder)

