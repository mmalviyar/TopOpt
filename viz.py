import cv2
import numpy as np
import matplotlib.pyplot as plt

class Visualization:

	def __init__(self,gen,disc):

			self.arrow_length = 7
			self.gsize = (7,7)
			self.tol = 6
			self.gen = gen
			self.disc = disc
			self.training = False

	def generate_images(self,images):

			number_of_samples = len(images)

			if self.training is True:
				d = [200,300]
			else:
				d = [None,None]

			noise1 = tf.random.normal([number_of_samples, noise_dim], mean = 0.0, stddev = 1,seed = d[0])
			noise2 = tf.random.normal([number_of_samples, noise_dim], mean = 0.0, stddev = 1, seed = d[1])
	
			SIMPsol = images[:,:,:,0:1]
			test_input = images[:,:,:,1:6]

			gsol= gen([noise1,test_input],training = False)
			gsol1 = gen([noise2,test_input],training = False)
	

			idx = 0
			title = ['Problem','Ground Truth','Solution1','Solution2']

			for j in range(number_of_samples):
				img = images[j:j+1,:,:,:]
				
				display_list = [SIMPsol[j,:,:,0],gsol[j,:,:,0],gsol1[j,:,:,0]]

				for i in range(len(title)):          
					
					# if j == 0:
					#   plt.title(title[i-1])
						
					idx = idx + 1

					if i is 0:
						plt.subplot(number_of_samples, len(title), idx)
						self.vizProb(img)
					else:
						plt.subplot(number_of_samples, len(title), idx)
						plt.imshow(-display_list[i-1]*127.5 + 127.5, cmap = 'gray',vmin = 0 , vmax = 255)
						plt.axis('off')


	def vizProb(self,img):

			xlen = img.shape[1]
			ylen = img.shape[2]

			FC = img[0,:,:,3:5]    
			x,y,dx,dy = self.farrow(FC)

			image = np.ones(shape = (img.shape[1]+2*self.tol,img.shape[2]+2*self.tol,3))

			BCX = img[0,:,:,1:2]
			image[self.tol:xlen+self.tol,self.tol:ylen+self.tol,1] = cv2.GaussianBlur(BCX,self.gsize,cv2.BORDER_DEFAULT) + BCX[:,:,0]

			BCY = img[0,:,:,2:3]
			image[self.tol:xlen+self.tol,self.tol:ylen+self.tol,2] = cv2.GaussianBlur(BCY,self.gsize,cv2.BORDER_DEFAULT) + BCY[:,:,0]

			image[self.tol:xlen+self.tol,self.tol:ylen+self.tol,0] = img[0,:,:,5]

			image = np.array(127.5*image + 127.5)

			plt.imshow(image.astype(np.uint8))
			plt.arrow(x+self.tol,y+self.tol,dx,dy,head_length=3,head_width=1.5, color = 'black')
			plt.axis('off')
		
	def farrow(self,F):

			fx = F[:,:,0]
			fy = F[:,:,1]

			fmagX,locX = self.fnodes(fx)
			fmagY,locY = self.fnodes(fy)

			fmag = np.sqrt(fmagX*fmagX + fmagY*fmagY)
			dx = -self.arrow_length*(fmagX/fmag)
			dy = -self.arrow_length*(fmagY/fmag)

			if locX is None:
				loc = locY
			elif locY is None:
				loc = locX
			elif locX == locY:
				loc = locX
			else:
				loc = [[0],[0]]
		
			return loc[1][0],loc[0][0],dx,dy

	def fnodes(self,fc):
		
			fmax = np.max(fc)
			fmin = np.min(fc)
			fmid = np.median(fc)
			
			if abs(fmax-fmin) <= 0.001:
				fmag = 0
			elif abs(fmax-fmid) <= 0.001:
				fmag = fmin
			else:
				fmag = fmax
		 
			if fmag!= 0:
				loc = np.nonzero(fc==fmag)
				loc = loc[0],loc[1]   
			else:
				loc = [None,None]      
			return fmag,loc