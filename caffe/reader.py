# -*- coding: utf-8 -*-
"""
use to read the test label file
    then generate a file
"""
import shutil
import os,sys

bot2yu = {
    0: 4,
    1: 8,
    2: 7,
    3: 2,
    4: 1,
    5: 0,
    6: 10,
    7: 11,
    8: 3,
    9: 6,
    10: 5,
    11: 9
}

yu2category = {
    0: "wolf",
    1: "Dog",
    2: "fox",
    3: "giraffe",
    4: "guinea pig",
    5: "hyena",
    6: "reindeer",
    7: "sikadeer",
    8: "squirrel",
    9: "weasel",
    10: "cat",
    11: "chipmunk",
}

material_path = "/media/red/DATA/material/BOT/BOT-origin/"
test_path = "/media/red/DATA/material/BOT/"

def read(txt_path, count, dir):
    f = open(txt_path, 'r')

    while 1:
        lines = f.readlines(10000)
        if not lines:
            break
        for line in lines:
            msg = f.readline().split()
            if len(msg) == 3:
                classify(msg[0], int(msg[1]), dir)
                count += 1



def classify(file, label, dir):
    filename = file
    yu_label = bot2yu[label]
    category = yu2category[yu_label]

    _test_path = test_path + dir + "/"
    ori_path = _test_path + filename
    des_path = material_path + category + "/" + filename + "2"

    print(_test_path + filename)
    if os.path.exists(ori_path + ".jpg"):
        shutil.copyfile(ori_path + ".jpg", des_path  + ".jpg")
        print(file + ".jpg" + "==> " + category )
    elif os.path.exists(ori_path + ".png"):
        shutil.copyfile(ori_path  + ".png", des_path  + ".png")
        print(file + ".png" + "==> " + category )
    elif os.path.exists(ori_path + ".jpeg"):
        shutil.copyfile(ori_path  + ".jpeg", des_path + ".jpeg")
        print(file + ".jpeg" + "==> " + category )


if __name__ == '__main__':
    count = 0
    file = sys.argv[1]
    dir = sys.argv[2]
    read(file, count, dir)
    #print(count)
