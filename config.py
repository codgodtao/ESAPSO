# class FLAGES(object):
#     pan_size = 256
#
#     ms_size = 64
#     num_spectrum = 4
#
#     ratio = 4
#     stride = 16
#     norm = True
#
#     batch_size = 32
#     lr = 0.0001
#     decay_rate = 0.99
#     decay_step = 10000
#
#     # img_path='./data/source_data'
#     train_data_path = '/data/qlt/h5/full/training_data/train_wv4_data.h5'
#     valid_data_path = '/data/qlt/h5/full/validation_data/val_wv4_data.h5'
#     test_data_path = '/data/qlt/h5/test_data/WV4' \
#                      '/test_wv4_data_FR.h5'
#     log_dir = './log_11_25-generator_wv4'
#     model_save_dir = './model_11_25-generator_wv4'
#
#     is_pretrained = True
#
#     iters = 100000
#     model_save_iters = 500
#     valid_iters = 10
#     # iters = 50
#     # model_save_iters = 50
#     # valid_iters = 10
#     # model_path = "./model_11_25-generator_wv4/Generator-1000"
#     result_path = "./model_11_25-generator_wv4"

class FLAGES(object):
    pan_size = 256

    ms_size = 64

    num_spectrum = 4

    ratio = 4
    stride = 16
    norm = True

    batch_size = 32
    lr = 0.0001
    decay_rate = 0.99
    decay_step = 10000

    # img_path='./data/source_data'
    train_data_path = 'E:/data/train_wv4_data.h5'
    valid_data_path = 'E:/data/val_wv4_data.h5'
    test_data_path = 'E:\\UDL\\Data\\pansharpening\\' \
                     'test_data\\WV4\\test_wv4_data_FR.h5'
    log_dir = './log_11_25-generator'
    model_save_dir = './model_11_25-generator'

    is_pretrained = True

    iters = 100000
    model_save_iters = 500
    valid_iters = 10
    # iters = 50
    # model_save_iters = 50
    # valid_iters = 10
    model_path = "./model_11_25-generator/Generator-1000"
    result_path = "./model_11_25-generator"
