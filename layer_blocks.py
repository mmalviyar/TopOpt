
# IMPORTING python libraries
import numpy as np
import imageio
from tqdm import tqdm
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utility import utils
utils = utils()

class blocks:
	def __init__(self):
		self.SpFilters = 32
		 
	# Convolution blocks ----------Conv,Norm,Act,Pool,Dropout-----------------------------------------
	def Conv_block(self,x,filters,s = 3, act= 'leaky',scale = 'None',dropout = False,norm = 'batch'):

		x = utils.ConvLayer(x,filters,size=s)
		x = utils.Normalization(x,norm = norm)
		x = utils.activation(x,act =act)

		if scale is 'down':
			x = utils.DownPool(x)
		elif scale is 'up':
			x = utils.UpPool(x)
		else:
			x = x
		
		if dropout is True:
			x = layers.Dropout(0.2)(x)

		return x  

	# Residual blocks---------------------------------------------------------------------------------------
	def res_block(self,x,filters,current_filters,act = 'relu',no_of_layers = 2,norm = 'batch'):

		for i in range(no_of_layers):
			x = utils.Normalization(x,norm = norm)
			x = utils.activation(x,act = act)
			x = utils.ConvLayer(t,filters,size=s)

		
		if current_filters is not filters:
			z = utils.ConvLayer(t,filters,size=1)

		return x + z

	# Spatially Adaptive De Normalization Layer------------------------------------------------------------
	def SPADE(self,x,pb,filters,s = 3, norm = 'parafree',act = 'relu'):
		
		x = utils.Normalization(x,norm = norm)
		y = utils.ConvLayer(pb,self.SpFilters,size = s)
		y = utils.activation(y,act = act)
		
		gamma = utils.ConvLayer(x,filters,size = s)
		beta = utils.ConvLayer(x,filters,size = s)
				
		out = x*(1 + gamma) + beta        
		return out
	
	# SPADE residual blocks ------------------------------------------------------------------------------
	def SPADE_ResBlock(self,t,pb,filters,current_filters,s = 3, no_of_spade = 2, act = 'leaky', con = False):
		x = t
		fr = current_filters
		for i in range(no_of_spade):
			x = self.SPADE(x,pb,fr, s = s)
			x = utils.activation(x,act = act)
			if con == True:
				x = layers.Concatenate()([x,pb])
			x = utils.ConvLayer(x,filters,size =s)
			fr = filters
			
		if current_filters != filters:
			z = self.SPADE(t,pb,current_filters)
			z = utils.activation(t,act = act)
			z = utils.ConvLayer(t,filters,size=s)
		else:
			z = t
		
		return x + z

	# New Cells ---------------------------------

	def Coffee(self,t,pb,filters,norm = 'parafree',s = 3):
				
		x = utils.Normalization(t, norm = norm) # Dendrites

		alpha = utils.ConvLayer(x,filters,size = s, act = 'relu')
		x = utils.ConvLayer(x,filters,size = 1, act = 'leaky')
		y = utils.ConvLayer(pb,filters,size = s, act = 'relu')
		
		return alpha*x + y









