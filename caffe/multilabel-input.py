# -*- coding: utf-8 -*-

import numpy as np
import os
import lmdb
from PIL import Image 
import numpy as np 
import sys
# Make sure that caffe is on the python path:
caffe_root = 'your caffe root path'
sys.path.insert(0, caffe_root + '/python')
import caffe
####################train data(images)############################
# get imageFileList
file_list = os.listdir('your images path')
#your data lmdb path
in_db=lmdb.open('your data(images) lmdb path',map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx,in_ in enumerate(file_list):         
        im_file='your images path'+in_
        im=Image.open(im_file)
        im = im.resize((224,224),Image.BILINEAR)#根据网络resize VGGNet是222x224          
        im=np.array(im)
        im=im[:,:,::-1]#把im的RGB调整为BGR
        im=im.transpose((2,0,1))#把height*width*channel调整为channel*height*width
        im_dat=caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())   
        print 'data train: {} [{}/{}]'.format(in_, in_idx+1, len(file_list))        
        del im_file, im, im_dat
in_db.close()
print 'train data(images) are done!'
######train data of label################    
#txt with labels eg. (0001.jpg 2 5)
file_input=open('your label txt','r')
label1_list=[]
label2_list=[]
for line in file_input:
    content=line.strip()
    content=content.split(' ')
    label1_list.append(int(content[1]))
    label2_list.append(int(content[2]))
    del content
file_input.close() 
#your labels lmdb path
in_db=lmdb.open('your labels lmdb path',map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx,in_ in enumerate(file_list):
        target_label=np.zeros((2,1,1))# 2种lable
        target_label[0,0,0]=label1_list[in_idx]
        target_label[1,0,0]=label2_list[in_idx]
        label_data=caffe.io.array_to_datum(target_label)
        in_txn.put('{:0>10d}'.format(in_idx),label_data.SerializeToString())
        print 'label train: {} [{}/{}]'.format(in_, in_idx+1, len(file_list))
        del target_label, label_data    
in_db.close()
print 'train labels are done!'