import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from numba import jitclass, int32, float32
# TODO: Part2 of problem

# tuple dict for numba accelerator
spec = [
    ('param', float32[:]), ('size', int32), ('dt', float32), ('dx', float32), ('phi', float32), ('grid', float32[:, :]),
	('cgrid', float32[:, :]), ]


@jitclass(spec)
class CahnHil:
	"""	initialisation method with input of system variable, n(gird size), dt(time interval),
	dx(spatial interval), phi, var(tuple of parameters (a, M, k)"""
	def __init__(self, n, dt, dx, phi, var):
		self.param = var
		self.size = n
		self.dt, self.dx, self.phi = dt, dx, phi
		self.grid = np.full((n, n), phi, dtype=np.float32)
		self.cgrid = np.zeros((n, n), dtype=np.float32)

	# add random noise to the initial system
	def rand_noise(self):
		for i in range(self.size):
			for j in range(self.size):
				self.grid[i, j] = self.grid[i, j] + np.random.uniform(-0.1, 0.1)

	# determine periodic boundary conditions for the current input i,j co-ordinates of the gird
	def pbc(self, i, j, grid):
		near_neighbours = np.array([
			grid[i, (j - 1) % (self.size - 1)], grid[i, (j + 1) % (self.size - 1)],
			grid[(i + 1) % (self.size - 1), j], grid[(i - 1) % (self.size - 1), j]])
		return near_neighbours

	# update the order parameter of the system (1 sweep)
	def order_update(self):
		const = (self.dt * self.param[1]) / (self.dx ** 2)
		for i in range(self.size):
			for j in range(self.size):
				nn = self.pbc(i, j, self.cgrid)
				self.grid[i, j] = self.grid[i, j] + const * (np.sum(nn) - 4 * self.cgrid[i, j])
		return self.grid

	# update the chemical potenrial wrt to the order parameter
	def chem_pot(self):
		for i in range(self.size):
			for j in range(self.size):
				nn = self.pbc(i, j, self.grid)
				self.cgrid[i, j] = -self.param[0] * self.grid[i, j] + self.param[0] * (self.grid[i, j]) ** 3 - ((
						self.param[1] / self.dx ** 2) * (np.sum(nn) - 4 * self.grid[i, j]))

		return self.cgrid

	def grad(self, i, j, grid):
		nn = self.pbc(i, j, grid)
		x_grad = (nn[3]-nn[2])/(2*self.dx)
		y_grad = (nn[1]-nn[0])/(2*self.dx)
		gradient = x_grad**2 + y_grad**2
		return gradient

	def free_energy(self):
		free = 0.0
		for i in range(self.size):
			for j in range(self.size):
				free += ((self.param[0]/-2.0)*self.grid[i, j]**2 + (self.param[0]/4.0)*self.grid[i, j]**4 +
						(self.param[2]/2.0)*self.grad(i, j, self.grid)
						)
		return free


def update(framenum, img, x):
	for i in range(500):
		x.chem_pot()
		x.order_update()

	img.set_data(x.grid)

	return img,


def main(ni):
	z = np.asarray([0.1, 0.1, 0.1], dtype=np.float32)
	phi = 0.
	dx = 1
	dt = 2
	n = 50
	print(z)

	if ni == 0:
		output = open('free_energy_data.txt', 'w')
		x = CahnHil(n, dt, dx, phi, z)
		x.rand_noise()
		energy = []
		time = []
		for i in range(30000):
			x.chem_pot()
			x.order_update()
			if i >= 500 and i % 500 == 0:
				energy.append(x.free_energy())
				time.append(i)
		for i in range(len(time)):
			output.write(str(time[i]) + '\t' + str(energy[i]) + '\n')
		plt.plot(time, energy)
		plt.title('Free energy density over time')
		plt.xlabel('Time (Sweeps)')
		plt.ylabel('Free energy density')
		plt.savefig('energy_density.png')
		plt.show()
		output.close()
	elif ni == 1:
		fig, ax = plt.subplots()
		x = CahnHil(n, dt, dx, phi, z)
		x.rand_noise()
		img = ax.imshow(x.grid, animated=True)
		fig.colorbar(img)
		ani = animate.FuncAnimation(fig, update, fargs=(img, x, ), frames=100, interval=1, blit=True)

		return ani
	return None


inti = 0
a = main(inti)
