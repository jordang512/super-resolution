import sys
import os
import numpy as np
from scipy import interpolate, misc
from PIL import Image

train_image_names = os.listdir('train_images/y/')
y_train = []
x_train_raw = []
x_train = []

for im in train_image_names:
	pic = Image.open("train_images/y/" + im)
	pixels = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0], 3)
	x_raw = misc.imresize(pixels, .25, interp="nearest", mode='RGB')
	x = misc.imresize(x_raw, 400, interp="nearest", mode='RGB')
	print(x_raw.shape, x.shape, pixels.shape)

	y_train.append(pixels)
	x_train_raw.append(x_raw)
	x_train.append(x)
	misc.toimage(x).save('train_images/interp/' + im)
	misc.toimage(x_raw).save('train_images/x/' + im)
	print("processed image", len(y_train),"of", len(train_image_names))

training_split = int(len(y_train) * 0.7)
np.save('y_train.npy', y_train[:training_split])
np.save('x_train_raw.npy', x_train_raw[:training_split])
np.save('x_train.npy', x_train[:training_split])

np.save('y_test.npy', y_train[training_split:])
np.save('x_test_raw.npy', x_train_raw[training_split:])
np.save('x_test.npy', x_train[training_split:])