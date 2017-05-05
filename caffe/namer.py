# -*- coding: utf-8 -*-
"""
use to generate the label file :
    python namer.py path
"""
import os
import sys
import random


def getdir(rootDir, files, class_num):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if os.path.isdir(path):
            getlist(path, files, class_num, lists)
            class_num += 1


def getlist(dir, files, class_num, dirname):
    list = []
    for lists in os.listdir(dir):
        #path = os.path.join(dir, lists)
        if (lists.endswith(".jpg") or lists.endswith(".png") or lists.endswith(".jpeg")) and ("副本" not in lists):
            list.append(dirname + "/" + lists)

    files.append(list)


def print2file(files, train_file_path, val_file_path, ratio):
    train_str = ''
    val_str = ''

    train_file = open(train_file_path, 'w')
    val_file = open(val_file_path, 'w')

    class_num = 0
    for _class in files:
        #present_num = 0
        #train_num = int((1-ratio) * len(_class))
        #train_num = int(ratio * len(_class))

        for photo in _class:
            #present_num += 1
            rnd = random.random()
            if rnd < ratio:
                #train_str += photo + ' ' + str(class_num) + '\n'
                val_str += photo + ' ' + str(class_num) + '\n'
            else:
                #val_str += photo + ' ' + str(class_num) + '\n'
                train_str += photo + ' ' + str(class_num) + '\n'
        class_num += 1

    train_file.write(train_str)
    train_file.close
    val_file.write(val_str)
    val_file.close


if __name__ == '__main__':
    files = []
    class_num = 0
    ratio = 0.2

    path = sys.argv[1]
    train_file_path = path + "/train.txt"
    val_file_path = path + "/val.txt"

    getdir(path, files, class_num)
    print2file(files, train_file_path, val_file_path, ratio)
