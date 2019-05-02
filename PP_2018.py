import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate


'''
Inital condition phi = 1 for |r| < R

and 0 for |r| > R
'''

class Fisher:

	def __init__(self,n,R,phi,x,t,cond):
		self.N = n*n
		self.size = n
		self.rad = R
		self.dx = x
		self.dt = t

		self.grid = np.zeros((n, n), dtype=np.float32)

		if cond == 0:
			for i in range(n):
				for j in range(n):
					r = self.mag_r(i, j)
					if r < self.rad:
						self.grid[i,j] = phi
					else:
						self.grid[i,j] = 0
		if cond == 1:
			for i in range(n):
				for j in range(n):
					r = self.mag_r(i, j)
					if r < self.rad:
						self.grid[i, j] = np.exp(-0.5*r)
					else:
						self.grid[i, j] = 0



	def nearest(self,i,j,grid):
		near_neighbours = np.array([
			grid[i, (j - 1) % (self.size)], grid[i, (j + 1) % (self.size)],
			grid[(i + 1) % (self.size), j], grid[(i - 1) % (self.size), j]])
		return near_neighbours

	def mag_r(self,i,j):

		x_centre = self.size/2
		y_centre = self.size/2

		magsq = (i - x_centre) ** 2 + (j - y_centre) ** 2
		mag = magsq**0.5

		return mag

	def phi_up(self):
		old = self.grid.copy()
		for i in range(self.size):
			for j in range (self.size):
				nn = self.nearest(i,j,old)
				self.grid[i,j] = np.sum(nn)/4 - (1/(self.dt)**3)*(np.sum(nn) - 4*np.sum(nn)/4) + np.sum(nn)/4*(1-np.sum(nn)/4)





def update(framenum, img, x):
	x.phi_up()
	print(x.grid)
	img.set_data(x.grid)

	return img,


n = 50
R = 5
x = 1
t = (x)
phi = 0.1
cond = 1


fig, ax = plt.subplots()
x = Fisher(n,R,phi,x,t,cond)
img = ax.imshow(x.grid, animated=True)
fig.colorbar(img)
ani = animate.FuncAnimation(fig, update, fargs=(img, x, ), frames=1000, interval=1, blit=True)
plt.show()



