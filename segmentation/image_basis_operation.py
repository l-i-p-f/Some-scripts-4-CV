from skimage import io
from PIL import Image as img
import numpy as np

#Image和skimage读图片
img_pil = img.open('image/heart.png')       # PIL类型，对象是PIL图片
img_skio = io.imread('image/heart.png')     # numpy格式，对象是numpy数组，读入的数据是float32 型的，范围是0-1
print(type(img_pil))        # <class 'PIL.PngImagePlugin.PngImageFile'>
print(type(img_skio))       # <class 'numpy.ndarray'>

# PNG图片有4通道。第4通道A为透明通道。控制图片透明效果。
print(img_pil.size)    # PIL data_type:(width,height)
print(img_pil.mode,img_pil.format)    # mode = RGBA,format = PNG
# img_pil.show()
# img_pil.save('pil_save.png')

# ndarray数据格式没有show()和save()
print(img_skio.shape)   # np data_type:(height,width,channel)
print(img_skio.size)    # 像素个数
print(img_skio.max(),img_skio.min(),img_skio.mean())
# io.imsave('skio_save.png',img_skio)

# PIL与numpy格式转换
# PIL图片转换为numpy数组
im_array = np.array(img_pil)    # 也可以用np.asarray(im)，区别是np.array()是深拷贝，np.asarray()是浅拷贝
print(type(im_array))           # <class 'numpy.ndarray'>
io.imsave('pil2np_save.png',im_array)

# numpy数组转换为PIL图片
# 注意这里读入的数组是float32型的，范围是0-1，而PIL.Image 数据是uint8型的，范围是0-255，所以要进行转换
im_image = img.fromarray( np.uint8 ( img_skio * 255 ) )
print(type(im_image))       # <class 'PIL.Image.Image'>
img_pil.save('np2pil_save.png')
