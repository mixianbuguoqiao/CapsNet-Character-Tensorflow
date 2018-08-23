import os
import scipy
import numpy as np
import tensorflow as tf
import h5py
from PIL import Image
import progressbar
def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:

        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch

def load_character_data(batch_size, is_training=True):
    f=h5py.File("data/dataset.hdf5","r")
    X_train = (f['X_train'][()]/255.).astype(np.float32)
    Y_train = f['Y_train'][()].astype(np.int32)
    X_val = (f['X_val'][()] / 255.).astype(np.float32)
    Y_val = f['Y_val'][()].astype(np.int32)
    X_train = np.expand_dims(X_train,axis=-1)
    X_val = np.expand_dims(X_val,axis=-1)



    # X_train = image_pre_processing(X_train)
    # X_val = image_pre_processing(X_val)
    if is_training:
        trX, valX, trY, valY = X_train,X_val,Y_train,Y_val
        num_tr_batch = trX.shape[0]//batch_size
        num_val_batch = valX.shape[0]//batch_size
        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        X_test = (f['X_test'][()] / 255.).astype(np.float32)
        Y_test = f['Y_test'][()].astype(np.int32)
        num_te_batch = X_test.shape[0] // batch_size
        teX = X_test
        teY = Y_test
        return  teX, teY, num_te_batch

def image_pre_processing(x):
            noise = np.random.normal(loc=0.0,scale=0.1,size=x.shape)
            x = x*np.random.uniform(0.85,1.15) + np.random.normal(scale=0.2)  + noise
            x = np.clip(x,0.0,1.0)
            return x

def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'character':
        return load_character_data(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == 'character':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_character_data(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)


    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def load_test_dataset(test_dir, image_size):
    assert os.path.exists(test_dir)

    dirs = os.listdir(test_dir)
    final_vector_length = len(dirs)
    file_names = []

    X_train_list = []
    bar = progressbar.ProgressBar()
    for root, dirs, files in os.walk(test_dir + "/"):
        for image_file in files:
            file_names.append(image_file)
            im = Image.open(os.path.join(root, image_file))
            im = im.convert('L')
            im = im.resize((image_size, image_size))

            image_array = np.array(im)

            X_train_list.append(image_array)
    X_train = np.array(X_train_list).reshape((len(X_train_list), image_size, image_size))
    return X_train, file_names