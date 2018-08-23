import os
import scipy
import numpy as np
import tensorflow as tf
import h5py
import cv2
import argparse
from PIL import Image
from glob import glob
import skimage
def load_mnist():

        f = h5py.File("mnist.hdf5","w")
        path = os.path.join('data', 'mnist')

        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.uint8)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        f["X_train"] = trainX.reshape([60000,28,28])
        f["Y_train"] = trainY
        f.close()

def add_size(size_num):
        f = h5py.File("mnist.hdf5",'r')
        train = f["X_train"][()]
        num = train.shape[0]
        train = np.reshape(train,[num,28,28])

        #Image.fromarray(train[0]).show()
        result = np.zeros([num,28,28])

        allowance = int((28-size_num[0])/2)
        for i in range(num):

          term = cv2.resize(train[i],size_num,interpolation=cv2.INTER_CUBIC)
          term = np.pad(term, ((allowance, allowance), (allowance, allowance)), 'constant', constant_values=(0, 0))
          result[i] = term

        return result

def integeration(size):

          name = "mnist"+str(size[0])+".hdf5"
          f1 = h5py.File(name,"w")
          f2 = h5py.File("mnist.hdf5","r")
          result = add_size(size)
          train = result.astype(np.uint8)

          f1["X_train"] = train
          f1["Y_train"] = f2["Y_train"][()]

def grab_files(file_dir):
       file_dir+='/'
       return [f for f in glob(file_dir + '*') ]

def word_hdf5(file_dir,image_size):
       f = h5py.File("data64.hdf5", 'w')
       file_names = grab_files(file_dir)
       if len(file_names) == 0:
           return None
       file_term = []
       image_term = []
       index_term = []
       i = 0
       for file in file_names:
           i = i+ 1
           print(i)
           file_name = file[-1]
           file_term.append(file_name)

           pictures = grab_files(file)
           for picture in pictures:
               index_term.append(i)

               im = Image.open(picture).convert("RGB")
               term = np.array(im)
               term = cv2.resize(term,(image_size,image_size))

               image_term.append(term)

       file1 = np.array(file_term).reshape([100])
       image = np.array(image_term).reshape([40000,64,64,1]).astype(np.uint8)
       index = np.array(index_term).reshape([40000]).astype(int)
       print(file1)
       f["X_train"] = image
       f["Y_train"] = index
      # f["character_names"] = file1
       f.close()


def caltech(file_dir, image_size):
    f = h5py.File("caltech.hdf5", 'w')
    file_names = grab_files(file_dir)
    if len(file_names) == 0:
        return None
    file_term = []
    image_term = []
    index_term = []
    i = 0
    for file in file_names:
        i = i + 1
        print(i)
        file_name = file.split("\\")[-1]
        file_term.append(file_name)

        pictures = grab_files(file)
        for picture in pictures:
            index_term.append(i)

            im = Image.open(picture).convert("RGB")
            term = np.array(im)
            term = cv2.resize(term, (image_size, image_size))
            if term.shape != (64,64,3):
                print(file_name)
            image_term.append(term)
    file_num  = len(file_term)
    image_num = len(image_term)
    index_num = len(index_term)

    file1 = np.array(file_term).reshape([file_num])
    image = np.array(image_term)
    index = np.array(index_term).reshape([index_num]).astype(np.uint8)
    #print(file1)
    f["X_train"] = image
    f["Y_train"] = index
#    f["character_names"] = file1
    f.close()

