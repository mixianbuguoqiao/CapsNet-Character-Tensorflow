import h5py
from PIL import Image, ImageChops
import random
import matplotlib.pyplot as pyplot
import numpy as np
import tensorflow as tf
from scipy import misc
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
from utils import load_mnist
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import cv2
def make_data():
    f = h5py.File("dataset.hdf5",'r')

    f1 = h5py.File("data.hdf5","w")

    train_term  = np.zeros([35000,32,32,1],dtype="float32")
    val_term = np.zeros([2500,32,32,1],dtype = "float32")
    test_term = np.zeros([2500,32,32,1],dtype = "float32")
    Y_train_term  = np.zeros([35000,1],dtype="float32")
    Y_val_term = np.zeros([2500,1],dtype = "float32")
    Y_test_term = np.zeros([2500,1],dtype = "float32")
    for i in range(100):

        train_start = 400 * i
        train_end  = train_start + 350
        new_train_start = 350 * i
        new_train_end = new_train_start + 350

        train_term[new_train_start:new_train_end] = f["X_train"][()][train_start:train_end].reshape([350,32,32,1])
        Y_train_term[new_train_start:new_train_end] = f["Y_train"][()][train_start:train_end]

        val_start = train_end
        val_end = train_end + 25
        new_val_start = 25 * i
        new_val_end = new_val_start + 25
        val_term[new_val_start:new_val_end] = f["X_train"][()][val_start:val_end].reshape([25,32,32,1])
        Y_val_term[new_val_start:new_val_end] = f["Y_train"][()][val_start:val_end]

        test_start = val_end
        test_end = val_end + 25
        new_test_start = 25 * i
        new_test_end = new_test_start + 25
        test_term[new_test_start:new_test_end] = f["X_train"][()][test_start:test_end].reshape([25,32,32,1])
        Y_test_term[new_test_start:new_test_end] = f["Y_train"][()][test_start:test_end]
        print(i)
    Y_train_term = np.reshape(Y_train_term,[35000])
    Y_val_term = np.reshape(Y_val_term,[2500])
    Y_test_term  = np.reshape(Y_test_term,[2500])

    f1["X_train"] = train_term
    f1["Y_train"] = Y_train_term
    f1["X_val"] = val_term
    f1["Y_val"] = Y_val_term
    f1["X_test"] = test_term
    f1["Y_test"] = Y_test_term
    f1["character_names"] = f["character_names"][()]


def make_data1(file,image_size):
    f = h5py.File(file, 'r')

    f1 = h5py.File("data64train_val.hdf5", "w")
    f2 = h5py.File("dataset.hdf5","r")
    train_term = np.zeros([39000,image_size,image_size, 1], dtype=np.uint8)
    val_term = np.zeros([1000,image_size, image_size, 1], dtype=np.uint8)


    Y_train_term = np.zeros([39000], dtype=np.uint8)
    Y_val_term = np.zeros([1000], dtype=np.uint8)

    for i in range(100):
        train_start = 400 * i
        train_end = train_start + 390
        new_train_start = 390 * i
        new_train_end = new_train_start + 390

        train_term[new_train_start:new_train_end] = f["X_train"][()][train_start:train_end].reshape([390, image_size, image_size, 1])
        Y_train_term[new_train_start:new_train_end] = f["Y_train"][()][train_start:train_end]

        val_start = train_end
        val_end = train_end + 10
        new_val_start = 10 * i
        new_val_end = new_val_start + 10
        val_term[new_val_start:new_val_end] = f["X_train"][()][val_start:val_end].reshape([10,image_size,image_size, 1])
        Y_val_term[new_val_start:new_val_end] = f["Y_train"][()][val_start:val_end]


        print(i)



    f1["X_train"] = train_term.astype(np.uint8)
    f1["Y_train"] = Y_train_term.astype(np.uint8)
    f1["X_val"] = val_term.astype(np.uint8)
    f1["Y_val"] = Y_val_term.astype(np.uint8)


    f1["character_names"] = f2["character_names"][()]



# f1 = h5py.File("data96_train_val.hdf5", "r")
# X_train = f1["X_train"][()].reshape([39000,64,64])[1250]
# Y_train = f1["Y_train"][()][1250]
# Z_train = f1["character_names"][()][3]
# Image.fromarray(X_train).show()
# print(Y_train)
# print(Z_train)




def create_mini_data():
    f = h5py.File("data_size28",'r')
    minif = h5py.File("datamini28.hdf5","w")

    minif["X_train"] = f["X_train"][()].astype(np.uint8)[0:3500]
    minif["Y_train"] = f["Y_train"][()].astype(np.uint8)[0:3500]

    minif["X_val"] = f["X_val"][()].astype(np.uint8)[0:250]
    minif["Y_val"] = f["Y_val"][()].astype(np.uint8)[0:250]

    minif["X_test"] = f["X_test"][()].astype(np.uint8)[0:250]
    minif["Y_test"] = f["Y_test"][()].astype(np.uint8)[0:250]

    minif["character_names"] = f["character_names"][()][0:10]

    minif.close()
    f.close()
#create_mini_data()
#随机旋转图片
def random_rotate_image():
    f = h5py.File("more_data.hdf5","r")
    f1 = h5py.File("data_argument.hdf5","w")

    X_train = f["X_train"][()]
    X_val = f["X_val"][()]
    Y_train = f["Y_train"][()]
    Y_val = f["Y_val"][()]
    character_names = f["character_names"][()]

    train_num = X_train.shape[0]
    val_num = X_val.shape[0]

    X_train = X_train.reshape([train_num,32,32])
    X_val = X_val.reshape([val_num, 32, 32])


    train_term = np.zeros([39000 * 2, 32, 32, 1], dtype="float32")
    val_term = np.zeros([1000 * 2,32,32,1],dtype = "float32")
    y_train_term = np.zeros([39000 * 2], dtype="float32")
    y_val_term = np.zeros([1000 *2],dtype="float32")


    train_term[0:39000] = X_train.reshape([train_num,32,32,1])
    val_term[0:1000] = X_val.reshape([val_num,32,32,1])

    y_train_term[0:39000] = Y_train
    y_train_term[39000:78000] = Y_train
    y_val_term[0:1000] = Y_val
    y_val_term[1000:2000] = Y_val

    for i in range(train_num):
      term  = Image.fromarray(X_train[i]).rotate(15)
      term = np.array(term)
      train_term[ train_num + i ] = term.reshape([32, 32, 1])

    for j in range(val_num):

      term = Image.fromarray(X_val[j]).rotate(15)
      term = np.array(term)
      val_term[ val_num + j ] = term.reshape([32,32,1])

    f1["X_train"] = train_term
    f1["X_val"] = val_term
    f1["Y_train"] = y_train_term.astype(np.uint8)
    f1["Y_val"] = y_val_term.astype(np.uint8)
    f1["character_names"] = character_names
#随机偏移图片
def random_translate_image():
    f = h5py.File("more_data.hdf5","r")
    f1 = h5py.File("data_translate_argument.hdf5","w")

    X_train = f["X_train"][()]
    X_val = f["X_val"][()]
    Y_train = f["Y_train"][()]
    Y_val = f["Y_val"][()]
    character_names = f["character_names"][()]

    train_num = X_train.shape[0]
    val_num = X_val.shape[0]

    X_train = X_train.reshape([train_num,32,32])
    X_val = X_val.reshape([val_num, 32, 32])


    train_term = np.zeros([39000 * 2, 32, 32, 1], dtype = np.uint8)
    val_term = np.zeros([1000 * 2,32,32,1],dtype = np.uint8)
    y_train_term = np.zeros([39000 * 2], dtype = np.uint8)
    y_val_term = np.zeros([1000 *2],dtype = np.uint8)


    train_term[0:39000] = X_train.reshape([train_num,32,32,1])
    val_term[0:1000] = X_val.reshape([val_num,32,32,1])

    y_train_term[0:39000] = Y_train
    y_train_term[39000:78000] = Y_train
    y_val_term[0:1000] = Y_val
    y_val_term[1000:2000] = Y_val

    for i in range(train_num):
      xoff = random.randint(1,5)
      yoff = random.randint(1,5)
      term  = ImgOfffSet(X_train[i],xoff,yoff)

      term = np.array(term)
      train_term[ train_num + i ] = term.reshape([32, 32, 1])

    for j in range(val_num):
      xoff = random.randint(1, 5)
      yoff = random.randint(1, 5)
      term = ImgOfffSet(X_val[j],xoff,yoff)


      term = np.array(term)
      val_term[ val_num + j ] = term.reshape([32,32,1])

    f1["X_train"] = train_term
    f1["X_val"] = val_term
    f1["Y_train"] = y_train_term.astype(np.uint8)
    f1["Y_val"] = y_val_term.astype(np.uint8)
    f1["character_names"] = character_names


def ImgOfffSet(Img,xoff,yoff):
    Img = Image.fromarray(Img)

    c = ImageChops.offset(Img,xoff,yoff)
    # c.paste((0,0,0),(0,0,xoff,height))
    # c.paste((0,0,0),(0,0,width,yoff))
    return c


def adjust_size(size):
    f = h5py.File("data/dataset.hdf5","r")
    f1 = h5py.File("data_size28","w")

    X_train = f["X_train"][()]
    Y_train = f["Y_train"][()]

    X_val = f["X_val"][()]
    Y_val = f["Y_val"][()]

    X_test = f["X_test"][()]
    Y_test = f["Y_test"][()]



    character_names = f["character_names"][()]


    f1["Y_train"] = Y_train
    f1["Y_val"] = Y_val
    f1["Y_test"] =  Y_test
    f1["character_names"] = character_names


    train_term  = np.zeros([35000,size,size,1],dtype="float32")
    val_term = np.zeros([2500,size,size,1],dtype = "float32")
    test_term = np.zeros([2500,size,size,1],dtype = "float32")


    for train_index in range(35000):
        train = X_train[train_index].reshape([32,32])
        term = cv2.resize(train,(size,size)).reshape([size,size,1])
        train_term[train_index] = term

    for i in range(2500):
        val = X_val[i].reshape([32,32])
        term = cv2.resize(val,(size,size)).reshape([size,size,1])
        val_term[i] = term

        test = X_test[i].reshape([32,32])
        term1 = cv2.resize(test,(size,size)).reshape([size,size,1])
        test_term[i] = term1
    f1["X_train"] = train_term
    f1["X_val"] = val_term
    f1["X_test"] = test_term
    f1.close()


def adjust_size1(size):
    f = h5py.File("data/dataset.hdf5","r")
    f1 = h5py.File("data4000064","w")

    X_train = f["X_train"][()]
    Y_train = f["Y_train"][()]




    character_names = f["character_names"][()]


    f1["Y_train"] = Y_train

    f1["character_names"] = character_names


    train_term  = np.zeros([40000,size,size,1],dtype="float32")



    for train_index in range(40000):
        train = X_train[train_index].reshape([32,32])
        term = cv2.resize(train,(size,size)).reshape([size,size,1])
        train_term[train_index] = term
        print(train_index)

    f1["X_train"] = train_term

    f1.close()
def display(X,y):

    pca = decomposition.PCA(n_components=3)
    new_X = pca.fit_transform(X)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.nipy_spectral)
    plt.show()




data = h5py.File("data.hdf5","r")
X_train = data["X_train"][()].reshape([35000,32,32])
Y_train = data["Y_train"][()]
train_num = X_train.shape[0]


X_val = data["X_val"][()].reshape([2500,32,32])
Y_val = data["Y_val"][()]
val_num = X_val.shape[0]

X_test = data["X_test"][()].reshape([2500,32,32])
Y_test = data["Y_test"][()]
test_num = X_test.shape[0]

X_train_term = np.zeros([train_num,64,64,1],dtype= np.uint8)
X_val_term = np.zeros([val_num,64,64,1],dtype= np.uint8)
X_test_term = np.zeros([test_num,64,64,1],dtype= np.uint8)
for i in range(train_num):
    term = cv2.resize(X_train[i],(64,64))
    X_train_term[i] = term.reshape([64,64,1])
    print(i)
for j in range(val_num):
    term1 = cv2.resize(X_val[j],(64,64))
    term2 = cv2.resize(X_test[j],(64,64))

    X_val_term[j] = term1.reshape([64,64,1])
    X_test_term[j] = term2.reshape([64,64,1])
    print(j)
# X_val_test = np.zeros([val_num + test_num,64,64,1],dtype = np.uint8)
# for k in range(val_num + test_num):
#     X_val_test[k] =
testX = np.concatenate((X_val_term,X_test_term),axis=0)
testY = np.concatenate((Y_val,Y_test),axis=0)
print(testX.shape)
datafile = h5py.File("data3500064.h5","w")
datafile["X_train"] = X_train_term
datafile["Y_train"] = Y_train
datafile["X_test"] = testX
datafile["Y_test"] = testY
datafile.close()
