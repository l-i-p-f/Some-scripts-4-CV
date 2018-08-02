import os

imagefile = 'spatial_envelope_256x256_static_8outdoorcategories'
imagelist = os.listdir(imagefile)

imagenums = len(imagelist)

train_part = int(imagenums * 0.6)
val_part = int(imagenums * 0.2)
test_part = int(imagenums * 0.2)


def write_txt():
    with open('train.txt', 'w') as f:
        for i in range(0, train_part):
            f.writelines('/{}/{} /2dseg_for_deeplab/{}.png\n'.format(imagefile, imagelist[i], imagelist[i][:-4]))

    with open('test.txt', 'w') as f:
        for i in range(train_part, train_part + test_part):
            f.writelines('/{}/{} /2dseg_for_deeplab/{}.png\n'.format(imagefile, imagelist[i], imagelist[i][:-4]))

    if val_part == 0:
        print('Complete:\n  train.txt\n  test.txt\n')
        return

    with open('val.txt', 'w') as f:
        for i in range(imagenums - val_part, imagenums):
            f.writelines('/{}/{} /2dseg_for_deeplab/{}.png\n'.format(imagefile, imagelist[i], imagelist[i][:-4]))

    print('Complete:\n  train.txt\n  test.txt\n  val.txt\n')


write_txt()
