from scipy.spatial import Delaunay
import numpy as np

def sphere_triangulation(t_res, dl):
	#Creates graph for neural network
	dt = np.pi/t_res
	t_range=np.arange(-np.pi/2, np.pi/2, dt)
	t_range = np.append(t_range, np.pi/2)
	points=[]
	equivalents = []

	#Creating set of points for initial triangulation
	for t in t_range:
		if np.abs(np.cos(t))<1e-5:
			f_range = [0]
		else:
			df = dl/np.cos(t)
			f_range = np.arange(0, 2*np.pi*365.5/366, df)
		for f in f_range:
			points.append([t, f])
		equivalents.append([len(points)])
	equivalents.pop().pop()
	
	#Creating points for "gluing" the sphere along latitudes
	cover_points = [[t, np.pi*2] for i, t in enumerate(np.array(t_range))]
	cover_points.pop(0)
	for cover_point in cover_points:
		points.append(cover_point)
	points = np.array(points)

	#Triangulation process
	tri = Delaunay(points, incremental=True)
	simplices = tri.simplices
	
	#Gluing the sphere
	for n, equiv in enumerate(equivalents):
		simplices[simplices == len(points) - len(equivalents) + n] = equiv
	points = points[:-len(equivalents)+1]
	return points, simplices
