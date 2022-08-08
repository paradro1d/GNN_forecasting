import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata

def grid_to_nodes(nodes_coords, grid_data, res=2):
    #Grid_data : (lats, longs, values)
    _grid_data = np.transpose(grid_data, axes=[2, 0, 1])
    nodes_deg = nodes_coords*180/np.pi/res
    nodes_deg[:, 0] = nodes_deg[:, 0] + 90/res
    nd_interp = lambda x: map_coordinates(x, nodes_deg.T, order=1, mode='wrap')
    interp_fun = np.vectorize(nd_interp, signature='(n, m)->(k)')
    interpolated = interp_fun(_grid_data)
    interpolated = interpolated.T
    return interpolated

def nodes_to_grid(nodes_data, nodes_coords, res=2):
    #Nodes_data : (node_number, pres_lvls, values)
    nodes_deg = nodes_coords*180/np.pi
    nodes_deg[:, 0] = nodes_deg[:, 0] + 90
    wrapping = [nodes_deg, 
            nodes_deg + [0, 360], 
            nodes_deg - [0, 360], 
            nodes_deg * [-1, 1], 
            nodes_deg * [-1, 1] + [360, 0]]
    wrapped_coords = np.concatenate(wrapping, axis=0)
    wrapping = [nodes_data for _ in range(5)]
    wrapped_data = np.concatenate(wrapping, axis=0)
    lats, lons = np.mgrid[0:181:res, 0:360:res]
    interpolated = griddata(wrapped_coords, wrapped_data, (lats, lons), method='linear')
    return interpolated

def interpolation_error(nodes_coords, grid_data, res=2):
    interpolated = grid_to_nodes(nodes_coords, grid_data, res=res)
    reverse_interpolation = nodes_to_grid(interpolated, nodes_coords, res=res)
    lats = np.arange(0, 181, res)/180*np.pi - np.pi/2
    lats = np.cos(lats.reshape((-1, 1, 1)))
    return (((grid_data - reverse_interpolation)*lats)**2).mean()