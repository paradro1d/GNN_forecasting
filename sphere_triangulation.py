from scipy.spatial import Delaunay
import numpy as np
def sphere_triangulation(t_res, dl):
	dt = np.pi/t_res
	t_range=np.arange(0, np.pi + dt, dt)
	points=[]
	equivalents = []
	for t in t_range:
		if np.abs(np.sin(t))<0.00000001:
			f_range = [0]
		else:
			df = dl/np.sin(t)
			f_range = np.arange(0, 2*np.pi, df)
		for f in f_range:
			points.append([t, f])
		equivalents.append([len(points)])
	equivalents.pop().pop()
	cover_points = [[t, 2*np.pi] for t in t_range]
	cover_points.pop()
	cover_points.pop(0)
	for cover_point in cover_points:
		points.append(cover_point)
	points = np.array(points)
	tri = Delaunay(points)
	simplices = tri.simplices
	for n, equiv in enumerate(equivalents):
		simplices[simplices == len(points) - len(equivalents) + n] = equiv
	points = points[:-len(equivalents)+1]
	return points, simplices
