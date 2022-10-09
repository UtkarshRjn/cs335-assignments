from utils import *
import numpy as np


############ QUESTION 3 ##############
class KR:
	def __init__(self, x,y,b=1):
		self.x = x
		self.y = y
		self.b = b
	
	def gaussian_kernel(self, z):
		'''
		Implement gaussian kernel
		'''

		return np.exp(-np.square(z)/2)/np.sqrt(2*np.pi)

	def predict(self, x_test):
		'''
		returns predicted_y_test : numpy array of size (x_test, ) 
		'''

		x_test = x_test.reshape((x_test.shape[0],))
		self.x = self.x.reshape((self.x.shape[0],))

		N = self.x.shape[0]
		s = N * self.gaussian_kernel(np.subtract.outer(self.x,x_test,)/self.b) / np.sum(self.gaussian_kernel(np.subtract.outer(self.x, x_test)/self.b), axis=0)
		predicted_y_test = np.dot(s.T,self.y)/N
			

		return predicted_y_test
		
def q3():
	#Kernel Regression
	x_train, x_test, y_train, y_test = get_dataset()

	obj = KR(x_train, y_train)
	
	y_predicted = obj.predict(x_test)
	
	print("Loss = " ,find_loss(y_test, y_predicted))
