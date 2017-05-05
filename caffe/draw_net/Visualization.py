# --- Visualization.py ---
# -*- coding: utf-8 -*-

import caffe
import numpy as np
import matplotlib.pyplot as plt

# ！设置好路径
root = '/media/red/BRIDGE'
deploy = '/root/caffe-inland/models/bvlc_googlenet/deploy.prototxt'  # deploy文件
caffe_model = root + '/models/BOT/quick_quick_iter_760000.caffemodel'  # 训练好的 caffemodel
labels_filename = root + '/data/BOT/label.txt'  # 类别名称文件，将数字标签转换回类别名称
npy_mean_file = root + '/data/BOT/mean_file.npy'  # numpy格式的均值文件
test_image = root + '/material/BOT/test/cat.jpg'  # 测试图片所在的文件路径

# ！选择GPU：0-1080 1-750ti
caffe.set_device(1)
caffe.set_mode_gpu()

# ！加载网络
net = caffe.Net(deploy, caffe_model, caffe.TEST)
# 查看原始数据格式要求
# print 'input data:' + str(net.blobs['data'].data.shape)

# ！打开图像
im = caffe.io.load_image(test_image)
# 输出测试图片的数据信息
print '\ninput_img:'
print im.shape
# plt.imshow(im)
# plt.axis('off')

# ！图像处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(npy_mean_file).mean(1).mean(1))  # 减去均值
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
net.blobs['data'].data[...] = transformer.preprocess('data', im)
inputData = net.blobs['data'].data

# ！启动网络
net.forward()


# ！显示网络参数
def show_netdata():
    # 显示各层参数和形状
    print '\ndata_info of each layer:'
    data_list = [(k, v.data.shape) for k, v in net.blobs.items()]
    for i in data_list:
        print i
    # 输出网络参数
    print '\nparams_info of each layer:'
    params_list = [(k, v[0].data.shape) for k, v in net.params.items()]
    for i in params_list:
        print i


# ！输出网络结构
def show_data(data, output_name):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # 数据归一化
    data = (data - data.min()) / (data.max() - data.min())

    # 强制开方过滤器的数量
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters  
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)  
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)  

    # 输出权重到图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.figure()
    plt.imshow(data, cmap='gray')
    plt.axis('off')
    plt.savefig(root + '/data/BOT/visual/' + output_name + '.png', format='png')


def show_feature():
    # 以10*10大小显示图片，图形的插值采取就近原则，图像颜色是灰色
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    show_data(net.blobs['conv1/7x7_s2'].data[0], 'conv1_blobs')
    # print net.blobs['conv1/7x7_s2'].data.shape
    show_data(net.params['conv1/7x7_s2'][0].data.transpose(0, 2, 3, 1), 'conv1_params')
    # print net.params['conv1/7x7_s2'][0].data.shape
    show_data(net.blobs['pool1/3x3_s2'].data[0], 'pool1_blobs')
	# print net.blobs['pool1/3x3_s2'].data.shape

# ！显示减去均值前后的图片
def show_contrast():
    plt.figure()
    plt.subplot(1, 2, 1), plt.title("origin")
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1, 2, 2), plt.title("subtract mean")
    plt.imshow(transformer.deprocess('data', inputData[0]))
    plt.axis('off')
    plt.savefig(root + '/data/BOT/visual/contrast.png', format='png')


# ！显示分类结果
def get_label():
    f = open(labels_filename)
    label = f.read()
    f.close()
    labels = label.split()
    # print labels
    return labels

# 饼图
def draw_pie(prob):
    plt.figure(1)
    plt.axes(aspect=1)
    plt.title('Test', size=14)
    plt.pie(prob, labels = get_label())
    plt.savefig(root + '/data/BOT/visual/plot1.png', format='png')

# 直方图
def draw_bar(prob):
    plt.figure(2)
    plt.title('Test', size=16)
    plt.xlabel('label', size=14)
    plt.ylabel('prob', size=14)
    plt.bar(np.arange(len(prob)), prob, width=0.5, color='green', align='center', yerr=0.000001)
    plt.xticks(np.arange(len(prob)), (get_label()))
    # plt.text(-9,100, r'666', size=16)
    plt.savefig(root + '/data/BOT/visual/plot2.png', format='png')

# 直线
def draw_line(prob):
    plt.figure(3)
    plt.plot(prob)
    plt.title('Test', size=16)
    plt.savefig(root + '/data/BOT/visual/plot3.png', format='png')

# ！预测
def predict():
    labels = np.loadtxt(labels_filename, str, delimiter='\t')  # 读取类别名称文件
    prob = net.blobs['prob'].data[0]
    order = prob.argsort()[-1]
    print '\npredict:'
    print prob
    print 'top-1 predict:%s acc:%8.6f\n' % (labels[order], prob[order])
    draw_pie(prob)

predict()
