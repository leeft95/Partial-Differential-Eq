import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from numba import jit,jitclass, int32, float32
from matplotlib import cm

spec = [
    ('param', float32[:]),
	('size', int32),
	('dt', float32),
	('dx', float32),
	('phi', float32),
	('grid', float32[:,:]),
	('cgrid', float32[:,:]),

	# an array field
]
@jitclass(spec)
class CahnHil:
	"""	initialisation method with input of system variable, n(gird size), dt(time interval),
	dx(spatial interval), phi, var(tuple of parameters (a, M, k)"""
	def __init__(self, n, dt, dx, phi, var):
		self.param = var
		self.size = n
		self.dt, self.dx, self.phi = dt, dx, phi
		self.grid = np.full((n, n), phi,dtype = np.float32)
		self.cgrid = np.zeros((n, n),dtype = np.float32)


	# add random noise to the initial system
	#@jit(nopython = True)
	def rand_noise(self):
		for i in range(self.size):
			for j in range(self.size):
				self.grid[i, j] = self.grid[i, j] + np.random.uniform(-0.1, 0.1)

	# determine periodic boundary conditions for the current input i,j co-ordinates of the gird
	#@jit(nopython = True)
	def pbc(self, i, j, grid):
		near_neighbours = np.array([
			grid[i, (j - 1) % (self.size - 1)], grid[i, (j + 1) % (self.size - 1)],
			grid[(i + 1) % (self.size - 1), j], grid[(i - 1) % (self.size - 1), j]])
		return near_neighbours

	# update the order parameter of the system (1 sweep)
	#@jit(nopython = True)
	def order_update(self):
		const = (self.dt * self.param[1]) / (self.dx ** 2)
		for i in range(self.size):
			for j in range(self.size):
				nn = self.pbc(i, j,self.cgrid)
				self.grid[i, j] = self.grid[i, j] + const *(np.sum(nn) - 4 * self.cgrid[i, j])
				#print(self.grid[i,j])
		return self.grid
	#@jit(nopython = True)
	def chem_pot(self):
		for i in range(self.size):
			for j in range(self.size):
				nn = self.pbc(i, j,self.grid)
				self.cgrid[i, j] = -self.param[0] * self.grid[i, j] + self.param[0] * (self.grid[i, j]) ** 3 - ((
						self.param[1] / self.dx ** 2) * (np.sum(nn) - 4 * self.grid[i, j]))

		return self.cgrid

#@jit(nopython = True)
def update(frameNum,img,x):
	for i in range(50):
		x.chem_pot()
		x.order_update()

	img.set_data(x.grid)
	#plt.imshow(x.grid)
	return (img,)


def main():
	z = np.asarray([0.1, 0.1, 0.5],dtype = np.float32)
	phi = 0.5
	dx = 1
	dt = 2
	n = 50
	print(z)
	fig, ax = plt.subplots()
	x = CahnHil(n, dt, dx, phi, z)
	x.rand_noise()
	img = ax.imshow(x.grid, animated = True, interpolation = 'nearest', cmap = cm.gnuplot )#, vmin = -0.1, vmax = 0.1)
	fig.colorbar(img)
	ani = animate.FuncAnimation(fig, update, fargs=(img, x, ),
	                            frames=100,
	                            interval = 1,
	                            blit=True)

	return ani

a = main()
#plt.show()
#plt.imshow(x.grid, vmax=1, vmin=0)
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.imshow(x.cgrid)
# plt.show()
