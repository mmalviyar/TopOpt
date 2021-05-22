# IMPORTING python libraries
import numpy as np
import imageio
from tqdm import tqdm
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class utils:

  def __init__(self):
      self.mean = 0.
      self.variance = 0.02
      self.init = tf.random_normal_initializer(self.mean, self.variance)
  
  # --------------------CONVLAYER----------------------------------
  def ConvLayer(self,x,filters,size = 3,Str = 1, pad = 'same'):
      # init = tf.random_normal_initializer(self.mean, self.variance)
      x = layers.Conv2D(filters,size,strides = Str, padding= pad,kernel_initializer = self.init)(x)
      return x
  
  # --------------------NORMALIZATION------------------------------
  def Normalization(self,x,norm = 'parafree'):      
      epsilon = 1e-5
      if norm is 'sync':
        n = tf.keras.layers.experimental.SyncBatchNormalization
        x = n()(x)
      elif norm is 'batch':
        x = layers.BatchNormalization()(x)
      elif norm is 'parafree':
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_std = tf.sqrt(x_var + epsilon)
        x = (x - x_mean)/x_std
      else:
        x = x
      return x

  # -----------------CONDITIONAL NORMALIZATION ------------
  
  def adain(self,content, style, epsilon=1e-5, data_format='channels_first'):
        axes = [2,3] if data_format == 'channels_first' else [1,2]

        c_mean, c_var = tf.nn.moments(content, axes=axes, keepdims=True)
        s_mean, s_var = tf.nn.moments(style, axes=axes, keepdims=True)
        c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

        return s_std*(content - c_mean)/c_std + s_mean

  #------------------------UPSCALE,DOWNSCALE,RESIZE---------------------

  def UpPool(self,x,t=2,method = 'BI'):
      if method is 'NN':
        return layers.UpSampling2D(t)(x)
      else:
        return layers.UpSampling2D(t, interpolation='bilinear')(x)

  def DownPool(self,x,t=2,method = 'Avg'):
      if method is 'Avg':
        return layers.AveragePooling2D(t)(x)
      elif method is 'Max':
        return layers.MaxPooling2D(t)(x)
      else:
        return print('Defined method does not exist')
    
  def Resize(self,x,sizex,sizey,method = 'BI', aspect = True):
      if method is 'BI': 
        m = tf.image.ResizeMethod.BILINEAR
      else: 
        m = tf.image.ResizeMethod.NEAREST
      return tf.image.resize(x, [sizex,sizey], method=m, preserve_aspect_ratio=aspect)

  #--------------------Activations----------------------------------
  def activation(self,x,act = 'leaky'):

      if act is 'leaky':
        return layers.LeakyReLU()(x) 
      elif act is 'relu':
        return tf.keras.activations.relu(x)
      elif act is 'sigmoid':
        return tf.keras.activations.sigmoid(x)
      elif act is 'tanh':
        return tf.keras.activations.tanh(x)
      else:
        return x

    #---------------------FC Connected Layers-------------------

  def FcLayer(self,x,out,act = 'leaky', norm = 'batch'):

      x = layers.Dense(out,kernel_initializer = self.init)(x)
      if norm is not None:
         x = self.Normalization(x,norm = norm)
      if act is not None:
         x = self.activation(x,act = act)   

      return x


  def FcLayer_Reshape(self,x,out,act ='leaky',norm = 'batch'):

      n_nodes = out[0]*out[1]*out[2]
      x = layers.Dense(n_nodes,kernel_initializer = self.init)(x)
      
      if norm is not None:
         x = self.Normalization(x,norm = norm)
      if act is not None:
         x = self.activation(x,act = act)

      return layers.Reshape((out[0],out[1],out[2]))(x)   

    #---------------------------------------------------END----------------------------------------------------