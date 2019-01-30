import numpy as np
import keras
import h5py
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization, Conv1D, Input, Flatten, Dropout, MaxPooling1D, Add, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras import regularizers

def layer_1(x):
    c_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                kernel_regularizer=regularizers.l2(regularization_strength)
                )(x)
    bn_1 = BatchNormalization()(c_1)
    a_1 = Activation('relu')(bn_1)
    return a_1, x

def residual_block(a, filter_count=64, subsample=False):
    c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                kernel_regularizer=regularizers.l2(regularization_strength)
                )(x)
    bn1 = BatchNormalization()(c1)
    r1 = Activation('relu')(bn1)
    d1 = Dropout(0.5)(r1)
    return d1

def final_layer(a, x):
	y = Add()([x, a])
	return y

def PSNR_loss(yTrue, yPred):
	residuals = yTrue - yPred
	return (np.sum(residuals**2)) / 2

x = Input(batch_shape=(None, None, None, None))
a, base = layer_1(x)
for k in range(1, 30):
    a = residual_block(a, filter_count=32*(k//8+1), subsample=k%2)
y = final_layer(a, base)
model = Model(x, y)
model.summary()

learning_rate_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=0,
    min_lr=0
)

model.compile(
    loss=PSNR_loss
    optimizer=keras.optimizers.Adam()
)


model_local_path = 'checkpoints/cnn'

checkpoint = ModelCheckpoint(
    model_local_path + datetime.datetime.now().strftime("%I_%M%p-%b-%d") + '.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)

x_train = np.load('x_train.npy')
# x_train = np.expand_dims(x_train, axis=2)
y_train = np.load('y_train.npy')

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    # validation_data=(x_validate, y_validate),
    callbacks=[learning_rate_callback, checkpoint]# , s3_callback]
)