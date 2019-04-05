import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jitclass,float64,int32,float32
import mayavi.mlab as m
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection = '3d')

spec = [
    ('potential', float32[:, :, :]), ('dimension', int32), ('rho', float32[:, :, :]),
	]

#@jitclass(spec)
class Poissions():

	def __init__(self, n):

		self.potential = np.zeros((n,n,n),dtype = np.float32)

		#self.potential[0,1,2] = 5
		print(self.potential[0,1,2])
		#print(self.potential)
		# self.rho = np.random.normal(0.0,1,(n, n, n))
		# print(self.rho)
		np.zeros((n, n, n),dtype = np.float32)
		h = n/(n-1)
		q = 1
		rho0 = q/(h ** 2)
		# print(rho0)
		self.rho = np.zeros((n,n,n),dtype = np.float32)
		self.rho[int(n / 2), int(n / 2), int(n/2)] = rho0
		# print(self.rho)

		self.dimension = n

	def nearest(self,i,j,k,grid):

		near = np.array([grid[(i+1), j, k],
		                 grid[(i-1), j, k],
		                 grid[i,(j+1), k],
		                 grid[i, (j-1), k] ,
		                 grid[i, j, (k+1)],
		                 grid[i, j, (k-1)]
		                 ])

		return near

	def rho_gauss(self):
		pot = self.potential.copy()
		error = 0
		for i in range(1,(self.dimension-1)):
			for j in range(1,(self.dimension-1)):
				for k in range(1,(self.dimension-1)):
					nn = self.nearest(i,j,k,pot)
					self.potential[i,j,k] = 1/6*(np.sum(nn) + self.rho[i,j,k])
					error += abs(self.potential[i,j,k] - pot[i,j,k])
		return error

	def rho_gauss_2(self,w):
		error = 0
		old = self.potential.copy()
		for i in range(1,self.dimension - 1):
			for j in range(1,self.dimension - 1):
				for k in range(1,self.dimension - 1):
					nn = self.nearest(i, j, k, old)
					nn1 = self.nearest(i,j,k,self.potential)
					self.potential[i, j, k] = (1-w)*self.potential[i,j,k] + w/ 6 * (nn1[1]+ nn[0] +
					                                                                nn1[3] + nn[2] +
					                                                                nn1[5] + nn[4] +
					                                                                self.rho[i, j, k])
					error += abs(self.potential[i, j, k] - old[i, j, k])
		return error


n = int(5)
runs = []


con = 1e-5
err = 2*con
iterations = 0
w = 0.5
print(w)

iterations = 0
o = Poissions(n)
while err > con:
	err1 = err
	err = o.rho_gauss_2(w)
	err /= float (o.dimension **2)
	if err1 == err:
		break
	iterations += 1
err = 2 * con
print(" iterations = ", iterations)
runs.append(iterations)



output = open('E_data_out.txt','w')
output.write('x\ty\tz\n')


# o = Poissions(n)
# iterations = 0
# err = 2*con
# while err > con:
# 	err1 = err
# 	err = o.rho_gauss()
# 	err /= float (o.dimension **2)
# 	# print(err)
# 	#print(y.potential)
# 	iterations += 1
# print(" iterations = ", iterations)
#
x_1 = np.arange(0, o.dimension, 1)
y_1 = np.arange(0, o.dimension, 1)
z_1 =  np.arange(0, o.dimension, 1)


X,Y,Z = np.meshgrid(x_1, y_1, z_1, indexing = 'ij')


ef = np.gradient(o.potential)
ef = np.asarray(ef)
ef = -1*ef
print(ef.shape)
print('\n')
print(ef[0][0,1,1])

xe = []
ye = []
ze = []

for i in range(len(ef[0])):
	for k in range(len(ef[0][0])):
		for j in range(len(ef[0][0,1])):
			output.write(str(ef[0][i,k,j]) + '\t' +str(ef[1][i,k,j])  + '\t' +str(ef[2][i,k,j]) + '\n')
			xe.append(ef[0][i,k,j])
			ye.append(ef[1][i,k,j])
			ze.append(ef[2][i,k,j])
output.close()



xx = np.arange(0, len(ef[0]), 1)
y = np.arange(0, len(ef[0]), 1)
z = np.arange(0, len(ef[0]), 1)
Xe,Ye,Ze = np.meshgrid(xx,y,z,indexing = 'ij')


m.figure()
m.quiver3d(Xe,Ye,Ze,ef[0],ef[1],ef[2],colormap = 'gnuplot')
m.title('E-field/potential - vector/scalar')
m.vectorbar()
m.axes(ranges = [0, n, 0, n, 0, n])

a = m.contour3d(X,Y,Z,o.potential)
m.scalarbar()
m.axes(ranges = [0, n, 0, n, 0, n])
m.show()

c = tuple([o.potential[:,0,0],o.potential[0,:,0],o.potential[0,0,:],1])









