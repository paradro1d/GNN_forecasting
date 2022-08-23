import h5py
import numpy as np
import tensorflow as tf
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter
from scipy.spatial import KDTree
from tqdm import tqdm
from random import shuffle

def interpolation(grid_points, data):
    #Interpolating data onto grid points
    grid_x, grid_y = np.mgrid[0:78, 0:78]
    grid_values = data[grid_points[:, 0], grid_points[:, 1]]
    interpolation = griddata(grid_points, grid_values, (grid_x, grid_y),method='linear', fill_value=data.mean(axis=(0, 1)))
    return interpolation

def find_best_mesh_hierarchical(data, n_iters, n_best_points=30):
    #Finds best mesh including hierarchical points
    points = np.indices((78, 78))[:, ::7, ::7].reshape((2, -1)).T

    for _ in range(n_iters):
        #Interpolation and errors finding
        interp = interpolation(points, data)
        errors = np.abs(data - interp)

        #Extracting most suitable extrema points
        max_pooled = maximum_filter(errors, size=(3, 3), mode='constant')
        potential_points = np.array(np.where(max_pooled == errors))
        best_ind = np.argpartition(errors[potential_points[0], potential_points[1]], -n_best_points)[-n_best_points:]

        new_points = potential_points[:, best_ind].T
        points = np.concatenate([points, new_points], axis=0)

    return points.astype(np.int32)

def get_edges(nodes):
    #Finding edges using knn with 9 neighbors
    kdtree = KDTree(nodes)
    _, neighbors = kdtree.query(nodes, k=9, workers=-1)
    edges = np.stack([np.stack([neighbors[:, 0] for _ in range(8)], axis=-1), neighbors[:, 1:]], axis=-1)
    edges = edges.reshape((-1 ,2))

    '''
    #Taking unique edges and creating 2-way connectivity
    senders = edges.max(axis=1)
    receivers = edges.min(axis=1)

    packed_edges = np.stack([senders, receivers], axis=-1)
    unique_edges = np.unique(packed_edges, axis=0)
    edges = np.concatenate([unique_edges, unique_edges[:, ::-1]], axis=0)
    '''

    return edges.astype(np.int32)

def preprocess_and_save(filename, convert_filename, timesteps=4380):
    #Calculate optimal mesh and edges and save them into seperate dataset
    with h5py.File(filename, 'r') as read, h5py.File(convert_filename, 'w') as write:

        nodes_dataset = write.create_dataset('nodes', (timesteps, 744, 2), dtype='i4')
        edges_dataset = write.create_dataset('edges', (timesteps, 5952, 2), dtype='i4')

        print('Processing dataset')
        for i in tqdm(range(timesteps)):
            data = read['map_data'][i][:78, :78, 0, 1]
            mesh = find_best_mesh_hierarchical(data, 20)

            nodes_dataset[i] = mesh
            edges_dataset[i] = get_edges(mesh)

def training_gen(data_filename, mesh_filename, timesteps=4380):
    frames = list(range(timesteps - 1))
    shuffle(frames)
    with h5py.File(data_filename, 'r') as data, h5py.File(mesh_filename, 'r') as mesh:
        for frame in frames:
            yield (tf.convert_to_tensor(data['map_data'][frame]), 
                    tf.convert_to_tensor(mesh['nodes'][frame]),
                    tf.convert_to_tensor(mesh['edges'][frame]),
                    tf.convert_to_tensor(mesh['nodes'][frame + 1]))

hier_indices = lambda x: tf.cast(tf.concat([tf.floor((x + 3)/7), 
    (x + 3 - tf.floor((x + 3)/7)*7)], axis=0), tf.int32)

@tf.function(reduce_retracing=True)
def training_pipeline(data, nodes, edges, next_nodes):
    hier_nodes = tf.vectorized_map(hier_indices, tf.cast(next_nodes, tf.float32))
    target = tf.scatter_nd(hier_nodes, tf.ones(tf.shape(hier_nodes)[0]), tf.constant([12, 12, 7, 7]))
    target = tf.reshape(target, (-1, 7, 7))
    target = tf.reshape(target, (tf.shape(target)[0], -1))
    node_features = tf.gather_nd(data, nodes)
    node_features = tf.reshape(node_features, (tf.shape(node_features)[0], -1))
    senders = tf.reduce_max(edges, axis=1)
    receivers = tf.reduce_min(edges, axis=1)
    packed_edges = tf.stack([senders, receivers], axis=1)
    packed_edges = tf.bitcast(packed_edges, tf.int64)
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    edges = tf.concat([unique_edges, unique_edges[:, ::-1]], axis=0)
    edge_features = tf.gather(nodes, edges)
    edge_features = tf.cast(edge_features[:, 0] - edge_features[:, 1], tf.float32)
    edge_features = tf.concat([tf.expand_dims(tf.linalg.norm(edge_features, axis=-1), 
                                              -1), edge_features], axis=-1)
    return edges, edge_features, node_features, target

def eval_gen(data_filename, mesh_filename, timesteps):
    with h5py.File(data_filename, 'r') as data, h5py.File(mesh_filename, 'r') as mesh:
        for frame in timesteps:
            yield (tf.convert_to_tensor(data['map_data'][frame]), 
                    tf.convert_to_tensor(mesh['nodes'][frame]),
                    tf.convert_to_tensor(mesh['edges'][frame]),
                    tf.convert_to_tensor(mesh['nodes'][frame + 1]))


hier_to_st = lambda x: tf.stack([x[0]*7 + x[2] + 3, x[1]*7 + x[3] + 3])

def target_to_nodes(target):
    target = tf.reshape(target, (12, 12, 7, 7))
    nodes = tf.where(tf.cast(tf.nn.sigmoid(target) > 0.6, tf.float32) == 1)
    nodes = tf.vectorized_map(hier_to_st, nodes)
    nodes = tf.gather(nodes, tf.where(tf.math.logical_and(
    tf.math.logical_and(nodes[:, 0] < 80, nodes[:, 0] > -1),
    tf.math.logical_and(nodes[:, 1] < 80, nodes[:, 1] > -1)))[:, 0])
    return nodes
