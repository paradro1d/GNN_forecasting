import numpy as np
from stripy import sTriangulation
import tensorflow as tf
import random

def _faces_to_edges(faces):
    #Converting mesh faces to edges
    edges = tf.concat([faces[:, 0:2], faces[:, 1:3], 
            tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)

    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)

    packed_edges = tf.bitcast(tf.stack([senders, 
                receivers], axis=1), tf.int64)
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)

    senders, receivers = tf.unstack(unique_edges, axis=1)
    return tf.stack([tf.concat([senders, receivers],axis=0), 
            tf.concat([receivers, senders], axis=0)])


def uniform_mesh(lat_res, lon_res):
    '''
    Creates uniform mesh with given resolution:
        lat_res, lon_res - number of nodes on equator
    '''
    dlat = np.pi/lat_res
    dlon = np.pi/lon_res

    lat_range = np.arange(-np.pi/2, np.pi/2, dlat)
    points = []

    #Creating set of points for initial triangulation
    for lat in lat_range:
        if np.abs(np.cos(lat))<1e-6:
            lon_range = [0]
        else:
            dr = dlon/np.cos(lat)
            lon_range = np.arange(0, 2*np.pi, dr)
            for lon in lon_range:
                points.append([lat, lon])

    #Triangulating
    #random.shuffle(points)
    nodes = np.array(points)
    faces = sTriangulation(nodes[:, 1], nodes[:, 0]).simplices
    edges = _faces_to_edges(tf.convert_to_tensor(faces, dtype=tf.int32))
    return tf.convert_to_tensor(nodes, dtype=tf.float32), edges

def random_sampled_mesh(res, sampling_probs, size=3000):
    '''
    Creates randomly sampled mesh on a grid with given distribution
    '''
    grid = np.indices((int(180/res) + 1, int(360/res))).reshape((2, -1)).T
    sampled_points = np.random.choice(np.arange(grid.shape[0]),
            size=size, replace=False,
            p=sampling_probs.reshape(-1))
    nodes = grid[sampled_points]*res*np.pi/180
    faces = sTriangulation(nodes[:, 1], nodes[:, 0]).simplices
    edges = _faces_to_edges(tf.convert_to_tensor(faces))
    return tf.convert_to_tensor(nodes), edges
