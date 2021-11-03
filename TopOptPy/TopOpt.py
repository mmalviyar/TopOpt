import numpy as np 
from matplotlib import pyplot as plt
from utils import *
# from .optimizer import optimizer
# from .constructer import constructer
# from .

class TopOpt:
	def __init__(self, config): # Idea is 
		self.resolution = [32,64]
		self.vf = config["Target_volume_fraction"] # Returns a gray image with mag = vf
		self.b_x, self.by = self.get_boundary_condition(config["BoundaryCondition"]) # Returns empty image with boundary condition as 1
		self.f_x,self.f_y = self.get_force(config["Forces"])
		self.K = self.get_stiffness_matrix(config["Young_Modulus"], config["Poisson_ratio"]) #E,nu

	def get_boundary_condition(self,bc):
		# Conditions
		bc = loadJSON("configs/boundary_conditions")[bc] # Get locations in coordinates form of [32,64]


	def get_force():

	def convert_to_image():

	def visualize_case():

	def get_stiffness():

	def get_volume_fraction():


	def evaluate(self,solution):
		
		vf = self.get_volume_fraction(solution)
		compliance = self.get_compliance(solution)


	def optimizer(self):


  	def FE_setup(self):

      self.E0 = 1
      self.E = 1
      self.Emin = 1e-9
      self.Emax = 1
      self.nu = 0.3
      self.penal = 3.0
      self.ndof = 2*(self.nelx+1)*(self.nely+1)
      self.KE = self.K
      
      self.edofMat=np.zeros((self.nelx*self.nely,8),dtype=int)
     
      for elx in range(self.nelx):
        for ely in range(self.nely):
          el = ely+elx*self.nely
          n1=(self.nely+1)*elx+ely
          n2=(self.nely+1)*(elx+1)+ely
          self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

      # Construct the index pointers for the coo format--------------------------------------------------
      self.iK = np.kron(self.edofMat,np.ones((8,1))).flatten()
      self.jK = np.kron(self.edofMat,np.ones((1,8))).flatten() 


  	def get_compliance(self,sol):

      # Boundary Conditions------------
      xPhys = sol[1:self.nely+1,1:self.nelx+1].T
      xPhys = xPhys.reshape(self.nely*self.nelx)
      xPhys = np.array(xPhys,dtype=float)
      
      fixed = BC
      free = np.setdiff1d(self.dofs,fixed)
 
      # Force ------------------
      f=np.zeros((self.ndof,1))
      u=np.zeros((self.ndof,1))
      f[FC[0],0] = Fmag[0]
      f[FC[1],0] = Fmag[1]
  
      #--------------------------------------------------------------
      ce = np.ones(self.nely*self.nelx)
      sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(xPhys)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
      Kconst = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
 
      # Remove constrained dofs from matrix
      K = Kconst[free,:][:,free]
      # Solve system 
      u[free,0]=spsolve(K,f[free,0])
 
      # Compliance
      ce[:] = (np.dot(u[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * u[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1)
      obj=((self.Emin+xPhys**self.penal*(self.Emax-self.Emin))*ce).sum()

      return obj

  	def get_stiffness_matrix(self, E, nu): ## Get Stiffness Matrix, E: Young Modulus, nu: Poision Ratio

  	
 
      k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])

      KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
      [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
      [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
      [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
      [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
      [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
      [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
      [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
      return (KE)
