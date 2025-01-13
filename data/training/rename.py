import os

input_path = 'right'
image_names = os.listdir(input_path)

added_text = 'right'

for image in image_names:
    print(image)
    src = os.path.join(input_path, image)
    print(src)
    dst_filename = added_text + image
    print(dst_filename)
    dst = os.path.join(input_path, dst_filename)
    os.rename(src, dst)