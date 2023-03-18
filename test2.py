import h5py
import tensorflow as tf
import os
import numpy as np
# import gdal
import cv2
from DataSet import DataSet
from PanGan_2 import PanGan
from config import FLAGES
import scipy.io as scio
import time
import os
import tifffile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''定义参数'''


# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_string('pan_size',
#                            default_value=256,
#                            docstring='pan image size')
# tf.app.flags.DEFINE_string('ms_size',
#                            default_value=64,
#                            docstring='ms image size')
# tf.app.flags.DEFINE_string('batch_size',
#                            default_value=1,
#                            docstring='img batch')
# tf.app.flags.DEFINE_string('num_spectrum',
#                            default_value=4,
#                            docstring='spectrum num')
# tf.app.flags.DEFINE_string('ratio',
#                            default_value=4,
#                            docstring='pan image/ms img')
# tf.app.flags.DEFINE_string('model_path',
#                            default_value='model_11_25-generator',
#
#                            docstring='pan image/ms img')
# tf.app.flags.DEFINE_string('test_path',
#                            default_value='./data/test_gt',
#                            docstring='test img data')
# tf.app.flags.DEFINE_string('result_path',
#                            default_value='./result',
#                            docstring='result img')
# tf.app.flags.DEFINE_string('norm',
#                            default_value=True,
#                            docstring='if norm')


def main(argv):
    FLAGES.batch_size = 1
    if not os.path.exists(FLAGES.result_path):
        os.makedirs(FLAGES.result_path)
    model = PanGan(FLAGES.pan_size, FLAGES.ms_size, FLAGES.batch_size, FLAGES.num_spectrum, FLAGES.ratio, 0.001, 0.99,
                   1000, False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGES.model_path)
        test_path = FLAGES.test_data_path
        f = h5py.File(test_path, 'r')
        for index in range(271):
            pan = f['pan'][index].reshape((1024, 1024, 1))
            ms = f['ms'][index].reshape((256, 256, 4))
            pan = (pan - 1023.5) / 1023.5
            ms = (ms - 1023.5) / 1023.5
            h, w, c = ms.shape
            ms = cv2.resize(ms, (4 * w, 4 * h), interpolation=cv2.INTER_CUBIC)

            start = time.time()
            PanSharpening, error, error2 = sess.run(
                [model.PanSharpening_img, model.g_spectrum_loss, model.g_spatial_loss],
                feed_dict={model.pan_img: pan.reshape((1, 1024, 1024, 1)),
                           model.ms_img: ms.reshape((1, 1024, 1024, 4))})
            # test1, test2, test3 = model.PanSharpening_img, model.g_spectrum_loss, model.g_spatial_loss
            PAN_img = tensor2img_4C(PanSharpening)  # uint8
            print(PAN_img.shape)
            save_img(PAN_img,
                     '{}/{}_sr.png'.format(FLAGES.result_path, index))
            PanSharpening = PanSharpening * 1023.5 + 1023.5
            PanSharpening = PanSharpening.squeeze()
            end = time.time()
            print(end - start)
            save_name = f'output_mulExm_{str(index)}.mat'
            save_path = os.path.join(FLAGES.result_path, str(save_name))

            # cv2.imwrite(save_path, PanSharpening)
            # img_write(PanSharpening,save_path)
            # PanSharpening=cv2.cvtColor(PanSharpening[:,:,0:3], cv2.COLOR_BGR2RGB)
            # cv2.imwrite(save_path, PanSharpening)
            # tifffile.imsave(save_path, PanSharpening)
            scio.savemat(save_path,
                         {"sr": PanSharpening})  # H*W*C
            print(str(index) + ' done.' + 'spectrum error is ' + str(error) + 'spatial error is ' + str(error2))


def tensor2img_4C(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    修改输入通道数量，由RGB到包含4个或8个通道的图像，但是只取其中可见光的三个通道[4,256,256]
    '''
    tensor = tensor.squeeze()
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = len(tensor.shape)
    if n_dim == 4:
        n_img = len(tensor)
    elif n_dim == 3:
        img_np = tensor  # HWC, RGB
    elif n_dim == 2 or n_dim == 1:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        # 保存需要的sr结果

    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    # if mode == 'gray':
    #     cv2.imwrite(img_path, img)
    # else:
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


# def read_img(pan_test_path, ms_test_path, img_name, FLAGS):
#     pan_img_path=os.path.join(pan_test_path, img_name)
#     ms_img_path=os.path.join(ms_test_path, img_name)
#     #pan_img=cv2.imread(pan_img_path, -1)
#     #pan_img=gdal_read(pan_img_path,'pan')
#     pan_img=read8bit(pan_img_path,'pan')
#     h,w=pan_img.shape
#     pan_img=pan_img.reshape((1,h,w,1))
#     #ms_img=cv2.imread(ms_img_path, -1)
#     #ms_img=gdal_read(ms_img_path,'ms')
#     ms_img=read8bit(ms_img_path,'ms')
#     h,w,c=ms_img.shape
#     ms_img=cv2.resize(ms_img,(4*w,4*h),interpolation=cv2.INTER_CUBIC)
#     h,w,c=ms_img.shape
#
#     # ms_img=np.array(ms_img)
#     # h,w,c=ms_img.shape
#     # ms_img=cv2.resize(ms_img,(4*w,4*h),interpolation=cv2.INTER_CUBIC)
#     ms_img=ms_img.reshape((1,h,w,c))
#     return pan_img, ms_img

# def gdal_read(path,name):
#     data=gdal.Open(path)
#     w=data.RasterXSize
#     h=data.RasterYSize
#     img=data.ReadAsArray(0,0,w,h)
#     if name == 'ms':
#         img=np.transpose(img,(1,2,0))
#     img=(img-1023.5)/1023.5
#     return img

# def read8bit(path, name):
#     if name == 'ms':
#         v = 'src'
#     else:
#         v = 'pan'
#     v = 'I'
#     # img=scio.loadmat(path)[v]
#     img = np.load(path)
#     img = (img - 127.5) / 127.5
#     return img


# def img_write(img_array,save_path):
#     datatype=gdal.GDT_UInt16
#     h,w,c=img_array.shape
#     driver=gdal.GetDriverByName('GTiff')
#     data=driver.Create(save_path, w, h, c, datatype)
#     for i in range(c):
#         data.GetRasterBand(i+1).WriteArray(img_array[:,:,i])
#     del data
if __name__ == '__main__':
    tf.app.run()
