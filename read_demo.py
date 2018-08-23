import os
import numpy as np
from PIL import Image
import h5py
import cv2

# Folder path
train_PATH = "./jaffe/t/"
ver_PATH = "./jaffe/v/"
FILE_FORMAT = (".tiff", ".jpg", ".png")


# Get first three digits
def getImageId(name):
    s = name[:]
    r = s[0:s.find(' ', 1)]
    return int(r)


# def translate(st):
#     if st == "KA":
#         return 0
#     elif st == 'KL':
#         return 1
#     elif st == 'KM':
#         return 2
#     elif st == 'KR':
#         return 3
#     elif st == 'MK':
#         return 4
#     elif st == 'NA':
#         return 5
#     else:
#         # print "\"%s\" is not an integer."%st
#         pass
#     return st


train_images = []
train_imagesResized = []
train_ethnic = []
ver_images = []
ver_imagesResized = []
ver_ethnic = []

for subdir, dirs, files in os.walk(train_PATH):
    for file in files:
        if file.endswith(FILE_FORMAT):
            name = os.path.join(subdir, file)
            im = cv2.imread(name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            # Image.fromarray(im).show()
            print(im.shape)
          #  im.show()
            train_images.append(np.array(im))

            im = cv2.resize(im, (32, 32))
           # print(im)
          #  Image.fromarray(im).show()
            train_imagesResized.append(np.array(im))
            imageId = getImageId(file)
            train_ethnic.append(imageId)
            # print(name)

# Concatenate
# images = np.float64(np.stack(images))
# print(images.shape)
train_imagesResized = np.float64(np.stack(train_imagesResized))
train_ethnic = np.stack(train_ethnic)

# Normalize data
# images /= 255.0

# Save to disk
f = h5py.File("train_images1.h5", "w")
# Create dataset to store images
# X_dset = f.create_dataset('data', images.shape, dtype='f')
# X_dset[:] = images
Xt_dset = f.create_dataset('train_dataResized', train_imagesResized.shape, dtype='f')
Xt_dset[:] = train_imagesResized
print(Xt_dset)
# Create dataset to store labels
yt_dset = f.create_dataset('train_ethnic', train_ethnic.shape, dtype='f')
yt_dset[:] = train_ethnic
f.close()

for subdir, dirs, files in os.walk(ver_PATH):
    for file in files:
        if file.endswith(FILE_FORMAT):
            name = os.path.join(subdir, file)
            im = cv2.imread(name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

            # im.show()
            ver_images.append(np.array(im))

            im = cv2.resize(im, (32, 32))
            ver_imagesResized.append(np.array(im))

            imageId = getImageId(file)
            ver_ethnic.append(imageId)
            # print(name)

ver_imagesResized = np.float64(np.stack(ver_imagesResized))
ver_ethnic = np.stack(ver_ethnic)



f1 = h5py.File("ver_images1.h5", "w")

Xv_dset = f1.create_dataset('ver_dataResized', ver_imagesResized.shape, dtype='f')
Xv_dset[:] = ver_imagesResized

yv_dset = f1.create_dataset('ver_ethnic', ver_ethnic.shape, dtype='f')
yv_dset[:] = ver_ethnic

f1.close()

# images = []
# imagesResized = []
# ethnic = []
#
# for subdir, dirs, files in os.walk(PATH):
#	for file in files:
#		if file.endswith(FILE_FORMAT):
#			name = os.path.join(subdir, file)
#			im = cv2.imread(name, cv2.IMREAD_COLOR)
#			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#			im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
#
#			# im.show()
#			images.append(np.array(im))
#
#			im = cv2.resize(im, (224, 224))
#			imagesResized.append(np.array(im))
#
#			imageId = getImageId(file)
#			ethnic.append(imageId)
#
## Concatenate
##images = np.float64(np.stack(images))
##print(images.shape)
# imagesResized = np.float64(np.stack(imagesResized))
# ethnic = np.stack(ethnic)
#
## Normalize data
##images /= 255.0
# imagesResized /= 255.0
## Save to disk
# f = h5py.File("images.h5", "w")
## Create dataset to store images
##X_dset = f.create_dataset('data', images.shape, dtype='f')
##X_dset[:] = images
# X_dset = f.create_dataset('dataResized', imagesResized.shape, dtype='f')
# X_dset[:] = imagesResized
#
## Create dataset to store labels
#
# y_dset = f.create_dataset('ethnic', ethnic.shape, dtype='f')
# y_dset[:] = ethnic
#
#
# f.close()
