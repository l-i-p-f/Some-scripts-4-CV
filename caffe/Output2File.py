# -*- coding: utf-8 -*-

import caffe
import numpy as np
import os

data = '/media/red/BRIDGE'
deploy = '/root/caffe-inland/models/bvlc_googlenet/deploy.prototxt'    # deploy文件
caffe_model = data + '/models/BOT/google_net_iter_200000.caffemodel'
labels_filename = data + '/data/BOT/label-yu.txt'    # 类别名称文件，将数字标签转换回类别名称
npy_mean_file = data + '/data/BOT/mean_file.npy'    # numpy格式的均值文件
test_file = data + '/material/BOT/Testset4/'  # 测试图片所在的文件路径
test_output = data + '/data/BOT/Result4_googLeNet_iter_200000.txt'   # 测试结果输出路径

# 选择GPU：0-1080 1-750
caffe.set_device(0)
caffe.set_mode_gpu()

toggle = {
    0:5,
    1:4,
    2:3,
    3:8,
    4:0,
    5:10,
    6:9,
    7:2,
    8:1,
    9:11,
    10:6,
    11:7
}

#---------- 获取图片名 ----------#

def getlist(dir, files, dirname):
    list = []
    for lists in os.listdir(dir):
        if lists.endswith(".jpg") or lists.endswith(".png") or lists.endswith(".jpeg"):
            list.append(dirname + lists)
    files.append(list)

files = []
getlist(test_file, files, test_file)

#---------- 测试图片 ----------#
test_str = ''
fp = open(test_output,'w')
for _class in files:
    for img in _class:
        net = caffe.Net(deploy,caffe_model,caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data',np.load(npy_mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

        im=caffe.io.load_image(img)
        net.blobs['data'].data[...] = transformer.preprocess('data',im)

        out = net.forward()

        labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
        prob= net.blobs['prob'].data[0].flatten() # 取出最后一层（prob）属于某个类别的概率值，并打印
        # print prob    # prob 存储的是每个分类的概率
        order1=prob.argsort()[-1]  # 将概率值排序，取出最大值所在的序号
        order2=prob.argsort()[-2]
	
       #  print 'top-1 predict:%s acc:%8.6f \ntop-2 predict:%s acc:%8.6f \n' %(labels[order1],prob[order1],labels[order2],prob[order2])   #将该序号转换成对应的类别名称，并打印
        test_str += img.split('/')[-1].split('.')[0] + '\t' + str(toggle[order1]) + '\t' + str(format(prob[order1], '.6f')) + '\t' + str(toggle[order2]) + '\t' + str(format(prob[order2], '.6f')) +'\n'

fp.write(test_str)
fp.close()
