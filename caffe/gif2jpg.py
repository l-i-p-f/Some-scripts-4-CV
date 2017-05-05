from PIL import Image
import os,sys

material_path = "/media/red/BRIDGE/material/BOT/BOT-origin/"

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
        if lists.endswith(".gif"):
            list.append(dirname + "/" + lists)

    files.append(list)


def gif2jpg(files):
    for _class in files:
        for file in _class:
            name = file[:-4]
            infile = material_path+name+".gif"
            outfile = material_path+name+".jpg"
            im = Image.open(infile)
            im = im.convert('RGB')
            im.save(outfile)
            print(infile, outfile)
            
if __name__ == '__main__':
    files = []
    class_num = 0
    ratio = 0.2

    path = sys.argv[1]
    train_file_path = path + "/train.txt"
    val_file_path = path + "/val.txt"

    getdir(path, files, class_num)
    gif2jpg(files)
