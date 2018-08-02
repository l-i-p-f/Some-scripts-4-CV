import numpy as np
import os
from PIL import Image

label_colours = {(0, 0, 0): 0,
                 (120, 120, 120): 1,  # awning
                 (180, 120, 120): 2,  # balcony
                 (6, 230, 230): 3,  # bird
                 (80, 50, 50): 4,  # boat
                 (4, 200, 3): 5,  # bridge
                 (120, 120, 80): 6,  # building
                 (140, 140, 140): 7,  # bus
                 (204, 5, 255): 8,  # car
                 (230, 230, 230): 9,  # cow
                 (4, 250, 7): 10,  # crosswalk
                 (224, 5, 255): 11,  # desert
                 (192, 65, 22): 12,  # door
                 (150, 5, 61): 13,  # fence
                 (120, 120, 70): 14,  # field
                 (0, 255, 0): 15,  # grass
                 (255, 6, 82): 16,  # moon
                 (143, 255, 140): 17,  # mountain
                 (204, 255, 4): 18,  # person
                 (255, 51, 7): 19,  # plant
                 (204, 70, 3): 20,  # pole
                 (0, 102, 200): 21,  # river
                 (128, 0, 255): 22,  # road
                 (255, 6, 51): 23,  # rock
                 (11, 102, 255): 24,  # sand
                 (0, 0, 255): 25,  # sea
                 (255, 9, 224): 26,  # sidewalk
                 (9, 7, 230): 27,  # sign
                 (0, 128, 255): 28,  # sky
                 (255, 9, 92): 29,  # staircase
                 (112, 9, 255): 30,  # streetlight
                 (255, 255, 0): 31,  # sun
                 (0, 128, 0): 32,  # tree
                 (128, 0, 0): 33  # window
                 }


def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in label_colours.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def main():
    new_label_dir = './labels_desnet/'
    if not os.path.exists(new_label_dir):
        os.mkdir(new_label_dir)

    label_dir = './siftflow_segmentation_image/'
    label_files = os.listdir(label_dir)

    for l_f in label_files:
        arr = np.array(Image.open(label_dir + l_f))
        arr_2d = convert_from_color_segmentation(arr)
        Image.fromarray(arr_2d).save(new_label_dir + l_f)
        print('.', end='')
    print('Convert segmentation image from 3d to 2d done.')


if __name__ == '__main__':
    main()
