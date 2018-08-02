import numpy as np
from skimage import io
import os

imagefile = 'spatial_envelope_256x256_static_8outdoorcategories/'
imagelist = os.listdir(imagefile)
save_path = 'image_mean.txt'


def save_mean(mean_r, mean_g, mean_b):
    with open(save_path, 'w') as f:
        f.write('mean_r:{}\nmean_g:{}\nmean_b:{}'.format(mean_r, mean_g, mean_b))


# 图片尺寸统一，且数量较少时使用
def compute_image_mean_sum():
    sum_r = sum_g = sum_b = 0.0
    div = len(imagelist) * 65536

    print('There are {} images needed to deal with.'.format(len(imagelist)))
    print('Computing image mean of all images...')

    for image in imagelist:
        img = io.imread(os.path.join(imagefile, image))
        sum_r += np.sum(img[:, :, 0])
        sum_g += np.sum(img[:, :, 1])
        sum_b += np.sum(img[:, :, 2])

    print('Done.')
    return sum_r / div, sum_g / div, sum_b / div


# 图片尺寸不统一时
def compute_image_mean_mean():
    sum_r = sum_g = sum_b = 0.0
    div = len(imagelist)
    print('There are {} pics needed to deal with.'.format(div))
    print('Computing mean of all images.')
    for image in imagelist:
        img = io.imread(os.path.join(imagefile, image))
        sum_r += np.mean(img[:, :, 0])
        sum_g += np.mean(img[:, :, 1])
        sum_b += np.mean(img[:, :, 2])
    print('Done.\n')

    return sum_r / div, sum_g / div, sum_b / div


def main():
    mean_r, mean_g, mean_b = compute_image_mean_sum()
    print('mean_red:', mean_r)
    print('mean_green:', mean_g)
    print('mean_blue:', mean_b)
    save_mean(mean_r, mean_g, mean_b)


if __name__ == '__main__':
    main()
