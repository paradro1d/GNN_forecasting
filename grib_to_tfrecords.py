import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from sphere_triangulation import sphere_triangulation
import tensorflow as tf
from tqdm import tqdm

def triangles_to_edges(faces):
	#Converting mesh faces to edges
	edges = tf.concat([faces[:, 0:2], faces[:, 1:3], tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
	receivers = tf.reduce_min(edges, axis=1)
	senders = tf.reduce_max(edges, axis=1)
	packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
	unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
	senders, receivers = tf.unstack(unique_edges, axis=1)
	return tf.stack([tf.concat([senders, receivers],axis=0), tf.concat([receivers, senders], axis=0)])

def grib_to_tfrecords(grib_filenames, long_res, lat_res, longs, lats):
	#Reading GRIB dataset
	GRIB_arrays = []
	#Creating graph on the sphere
	nodes, faces = sphere_triangulation(long_res, lat_res)

	for grib in grib_filenames:
		ds = xr.load_dataset(grib, engine='cfgrib')
		node_array = ds.to_array().to_numpy()
		#Flattening pressure levels
		node_array = np.transpose(node_array, axes=[1,3,4,2,0])
		node_array = node_array.reshape(*node_array.shape[:3], -1)

		interpolated = []
		#Interpolating given values onto mesh vertexes
		for frame in tqdm(node_array):
			interp = RGI((lats[::-1], longs), frame[::-1])
			interpolated.append(interp(nodes))
		node_array = np.stack(interpolated)
		GRIB_arrays.append(node_array)

	#Processing data to the input format
	node_features = tf.convert_to_tensor(np.concatenate(GRIB_arrays, axis=0))
	simplices = tf.convert_to_tensor(faces)
	edges = triangles_to_edges(faces)

	#Calculating edge features
	u_i = tf.transpose(tf.gather(nodes, edges[0]))
	u_j = tf.transpose(tf.gather(nodes, edges[1]))
	u_j = tf.stack([tf.cos(u_j[0])*tf.cos(u_j[1]), tf.cos(u_j[0])*tf.sin(u_j[1]), tf.sin(u_j[0])])
	u_i = tf.stack([tf.cos(u_i[0])*tf.cos(u_i[1]), tf.cos(u_i[0])*tf.sin(u_i[1]), tf.sin(u_i[0])])
	u_ij = u_i - u_j
	u_ij_norm = tf.expand_dims(tf.norm(u_ij, axis=0), 0)
	edge_features = tf.transpose(tf.concat([u_ij, u_ij_norm], axis=0))
	
	#Creating datasets
	nodes_ds = tf.data.Dataset.from_tensor_slices(node_features)
	edge_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(edges, 0))
	edge_feats_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(edge_features, 0))
	#Converting to tfrecords format
	nodes_ds = nodes_ds.map(tf.io.serialize_tensor, num_parallel_calls=4, deterministic=True)
	edge_ds = edge_ds.map(tf.io.serialize_tensor)
	edge_feats_ds = edge_feats_ds.map(tf.io.serialize_tensor)
	writer = tf.data.experimental.TFRecordWriter('nodes_features.tfrecords')
	writer.write(nodes_ds)
	writer = tf.data.experimental.TFRecordWriter('edges.tfrecords')
	writer.write(edge_ds)
	writer = tf.data.experimental.TFRecordWriter('edge_features.tfrecords')
	writer.write(edge_feats_ds)
