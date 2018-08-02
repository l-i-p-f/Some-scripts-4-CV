from skimage import io,transform,data
from PIL import Image as img
import numpy as np

#img_np和skimg_np读图片
img_np = io.imread('image/me.jpg')     # numpy格式，对象是numpy数组

roi = img_np[500:2000,300:2000,:]           # 裁剪
print(img_np.shape)
dst = transform.resize(img_np,(640,640))
print(type(dst))        # <class 'numpy.ndarray'>
# 显示
img_pil = img.fromarray(np.uint8 ( img_np ))
img_pil.show()