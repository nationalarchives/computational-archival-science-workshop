# Imports
from PIL import Image
import os, os.path


def image_stitch(path):
    imgs = []           # List for images
    valid_images = [".jpg",".gif",".png"]
    size = 128, 128     # Individual image resize

    #open file and read lines
    with open(path, 'r') as f:
        for line in f:      # If valid image open, resize, and add to list
            if any(word in line for word in valid_images):
                tmp = Image.open(line)
                tmp.thumbnail(size, Image.ANTIALIAS)
                imgs.append(tmp)
            else:           # If not a valid image pint error and exit
                print('not a valid image, exiting')
                exit()

    # Create new image
    result = Image.new('RGB', (128 * len(imgs), 128))
    # paste all images into one image
    for x in imgs:
        result.paste(im=imgs[x], box=(128 * x, 0))
    return result
