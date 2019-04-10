import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jitclass,float64,int32,float32
import mayavi.mlab as m
from mpl_toolkits.mplot3d import Axes3D


spec = [
    ('potential', float32[:, :, :]), ('dimension', int32), ('rho', float32[:, :, :]),
	]

@jitclass(spec)
class Poissions():

	def __init__(self, n):

		self.potential = np.zeros((n,n,n),dtype = np.float32)
		np.zeros((n, n, n),dtype = np.float32)
		h = n/(n-1)
		q = 1
		rho0 = q/(h ** 2)
		self.rho = np.zeros((n,n,n),dtype = np.float32)
		self.rho[int(n / 2), int(n / 2), int(n/2)] = rho0
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


while True:
	try:
		n = int(input('Input the integer dimension of the grid:\n'))
		if n <= 5:
			raise ValueError
		else:
			break
	except ValueError:
		print('Please input a valid grid size \n')

while True:
	try:
		calc = int(input('Choose field to calculate:\n 0 == E-Field\n 1 == B-Field  \n'))
		if calc <= -1 or calc > 1:
			raise ValueError
		else:
			break
	except ValueError:
		print('Please input a valid option\n')
while True:
	try:
		alg = int(input('Choose Algorithm:\n 0 == Gauss\n 1 == Jacobi \n -1 == Gauss find optimum omega \n'))
		if alg <= -1 or alg > 1:
			raise ValueError
		else:
			break
	except ValueError:
		print('Please input a valid option\n')

con = float(input('Enter value of error accuracy of the form 1e(x):\n'))


err = 2*con
iterations = 0

if alg == 0:
	iterations = 0
	o = Poissions(n)
	w = 2/(1+np.sin((np.pi/o.dimension+1)))
	while err > con:
		err1 = err
		err = o.rho_gauss_2(w)
		err /= float (o.dimension **3)
		print(err)
		if err1 == err:
			break
		iterations += 1
	print(" iterations = ", iterations)

elif alg == 1:
	o = Poissions(n)
	iterations = 0
	err = 2 * con
	while err > con:
		err1 = err
		err = o.rho_gauss()
		err /= float(o.dimension ** 3)
		# print(err)
		# print(y.potential)
		iterations += 1
	print(" iterations = ", iterations)
elif alg == -1:
	runs = []
	w = np.arange(0,2,0.01)
	for i in range(len(w)):
		iterations = 0
		o = Poissions(n)
		err = 2 * con
		while err > con:
			err1 = err
			err = o.rho_gauss_2(w[i])
			err /= float(o.dimension ** 3)
			iterations += 1
		print(" iterations = ", iterations)
		runs.append(iterations)
	plt.plot(w,runs)
	plt.title('Optimal value of SOR')
	plt.xlabel('$\omega$')
	plt.ylabel('Iterations')
	plt.savefig('SOR.png')
	plt.show()



if calc == 0:


	x_1 = np.arange(0, o.dimension, 1)
	y_1 = np.arange(0, o.dimension, 1)
	z_1 =  np.arange(0, o.dimension, 1)


	X,Y,Z = np.meshgrid(x_1, y_1, z_1, indexing = 'ij')


	ef = np.gradient(o.potential)
	ef = np.asarray(ef)
	ef = -1*ef

	np.savetxt('EField data.txt', ef, delimiter='\t', newline='\n', fmt='%s')
	np.savetxt('potField data.txt', o.potential, delimiter='\t', newline='\n', fmt='%s')

	xx = np.arange(0, len(ef[0]), 1)
	y = np.arange(0, len(ef[0]), 1)
	z = np.arange(0, len(ef[0]), 1)
	Xe,Ye,Ze = np.meshgrid(xx,y,z,indexing = 'ij')

	norm = o.dimension**3
	print(norm)
	ef = ef/norm
	m.figure()
	m.quiver3d(Xe,Ye,Ze,ef[0],ef[1],ef[2],colormap = 'gnuplot')
	m.title('E-field/potential - vector/scalar')
	m.vectorbar()
	m.axes(ranges = [0, n, 0, n, 0, n])

	m.figure()
	a = m.contour3d(X,Y,Z,o.potential/norm)
	m.scalarbar()
	m.axes(ranges = [0, n, 0, n, 0, n])
	m.show()



	d_values = []
	p_values = []
	half = int(n/2)
	for i in range(n):
	    for j in range(n):
	        for k in range(n):
	            distance = np.sqrt(((i-half)**2)+((j-half)**2)+((k-half)**2))
	            d_values.append(np.log(distance))
	            pot = o.potential[i,j,k]
	            p_values.append(np.log(pot))

	fit_x = []
	fit_y = []

	for i in range(len(d_values)):
		if d_values[i] <= 1.0:
			fit_x.append(d_values[i])
			fit_y.append(p_values[i])

	params = np.polyfit(fit_x, fit_y,1)

	plt.figure()
	plt.scatter(d_values, p_values, label = 'gradient = ' + str(params[1]))
	plt.xlabel('Distance from charge log(1/r)')
	plt.ylabel('Potential Values')
	plt.legend()
	plt.savefig('EPvD.png')


	output = open('EPvD data.txt','w')
	output.write('log D\tlog P\n')
	for i in range(len(d_values)):
		output.write(str(d_values[i]) + '\t' + str(p_values[i]) + '\n')
	output.close()

	dd_values = []
	e_values = []

	for i in range(len(Xe)):
		for j in range(len(Ye)):
			for k in range(len(Ze)):
				dist = np.sqrt(((i-half)**2)+((j-half)**2)+((k-half)**2))
				f = np.sqrt(ef[0,i,j,k]**2 + ef[1,i,j,k]**2 + ef[2,i,j,k]**2)
				if np.isinf(f) != True:
					dd_values.append(np.log(dist))
					e_values.append(np.log(f))

	fit_xx = []
	fit_yy = []


	for i in range(len(dd_values)):
		print(e_values[i])
		if dd_values[i] <= 1.0:
			fit_xx.append(dd_values[i])
			fit_yy.append(e_values[i])

	params = np.polyfit(fit_xx,fit_yy,1)

	output = open('EPvD data.txt','w')
	output.write('log D\tlog P\n')
	for i in range(len(dd_values)):
		output.write(str(dd_values[i]) + '\t' + str(e_values[i]) + '\n')
	output.close()


	plt.figure()
	plt.scatter(dd_values, e_values,label = 'gradient = ' + str(params[1]))
	plt.xlabel('Distance from charge log(1/r)')
	plt.ylabel('E-field Values')
	plt.legend()
	plt.savefig('EfvP.png')


	plt.show()

elif calc == 1:
	bi, bj = np.gradient(o.potential[:][:][1])
	print(o.potential[:][:][1])
	print('\n\n')
	print(bi)
	print('\n\n')
	print(bj)
	ef = np.array([-bi, bj, np.zeros((n, n))])
	normB = np.sqrt(ef[0] ** 2 + ef[1] ** 2)
	ef = ef / normB

	np.savetxt('BField data.txt', ef, delimiter='\t', newline='\n', fmt='%s')
	np.savetxt('vecField data.txt', o.potential[:][:][1], delimiter='\t', newline='\n', fmt='%s')

	plt.figure()
	plt.quiver(ef[0], ef[1], angles='xy')
	plt.imshow(o.potential[:][:][1])
	plt.xlabel('$B_{x}$ Field')
	plt.ylabel('$B_{y}$ Field')
	plt.title('Magnetic Field with Vector potential overlayed')
	plt.savefig('Bfield.png')

	d_values = []
	p_values = []
	half = int(n / 2)

	for i in range(n):
		for j in range(n):
			distance = np.sqrt(((i - half) ** 2) + ((j - half) ** 2))
			d_values.append(np.log(distance))
			pot = o.potential[i, j, int(n / 2)]
			if np.isinf(pot) != True:
				p_values.append(np.log(pot))

	plt.figure()
	plt.scatter(d_values, p_values)
	plt.xlabel('Distance from wire')
	plt.ylabel('Vector Potential')
	plt.savefig('VecpD.png')

	output = open('VecPvD data.txt', 'w')
	output.write('log D\tlog P\n')
	for i in range(len(d_values)):
		output.write(str(d_values[i]) + '\t' + str(p_values[i]) + '\n')
	output.close()







