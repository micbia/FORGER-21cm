import numpy as np

from keras import backend as K
from keras.losses import binary_crossentropy
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Layer, subtract



def R2(y_true, y_pred):
    ''' R^2 (coefficient of determination) regression score function. To avoid NaN, added 1e-8 to denominator ''' 
    SS_num =  K.sum(K.square(y_true-y_pred)) 
    SS_den = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_num/(SS_den + K.epsilon())


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 1))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def UnFreezeNetwork(net, state):
    for layer in net.layers:
        layer.trainable = state
    net.trainable = state



class NegativeLayer(Layer):
  def __init__(self):
    super(NegativeLayer, self).__init__()
  
  def call(self, inputs):
    ones_tensor = K.ones_like(inputs)
    negative_input = subtract([ones_tensor, inputs])
    return negative_input