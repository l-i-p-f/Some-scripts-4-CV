import numpy as np
from PIL import Image
import scipy.io
import os

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


def convert_from_segmentation_color(arr_2d):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in label_colours.items():
        m = np.all(arr_2d.reshape(arr_2d.shape[0], arr_2d.shape[1], 1) == i, axis=2)
        arr_3d[m] = np.array(c)

    return arr_3d


def main():
    filepath = './spatial_envelope_256x256_static_8outdoorcategories/'
    savepath = './siftflow_segmentation_image/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for mat in os.listdir(filepath):
        arr = scipy.io.loadmat(os.path.join(filepath, mat))
        seg = arr['S']
        image = convert_from_segmentation_color(seg)
        image = Image.fromarray(image)
        image.save(savepath + mat[:-3] + 'png')
        print('.', end='')
    print('Convert mat to segmentation image done.')


if __name__ == '__main__':
    main()
