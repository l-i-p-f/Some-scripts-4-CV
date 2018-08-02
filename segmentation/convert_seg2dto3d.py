'''
	Convert 2d segmentation image to 3d rgb image for visualization.
'''

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert 2d segmentation image to 3d rgb image")
    parser.add_argument("--image_file", type=str, default="", help="Path to the directory of 2d segmentation image")
    return parser.parse_args()


label_colours = {(0, 0, 0): 0,
                 # 0=background
                 (128, 0, 0): 1,
                 (0, 128, 0): 2,
                 (128, 128, 0): 3,
                 (0, 0, 128): 4,
                 (128, 0, 128): 5,
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (0, 128, 128): 6,
                 (128, 128, 128): 7,
                 (64, 0, 0): 8,
                 (192, 0, 0): 9,
                 (64, 128, 0): 10,
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (192, 128, 0): 11,
                 (64, 0, 128): 12,
                 (192, 0, 128): 13,
                 (64, 128, 128): 14,
                 (192, 128, 128): 15,
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                 (0, 64, 0): 16,
                 (128, 64, 0): 17,
                 (0, 192, 0): 18,
                 (128, 192, 0): 19,
                 (0, 64, 128): 20
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 }


def convert_from_segmentation_color(arr_2d):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in label_colours.items():
        m = np.all(arr_2d.reshape(arr_2d.shape[0], arr_2d.shape[1], 1) == i, axis=2)
        arr_3d[m] = np.array(c)

    return arr_3d


def main():
    args = get_arguments()
    image = Image.open(args.image_file)
    img = np.array(image)
    img = convert_from_segmentation_color(img)
    img = Image.fromarray(img)
    img.show()
    # img.save('save_path')


if __name__ == '__main__':
    main()
