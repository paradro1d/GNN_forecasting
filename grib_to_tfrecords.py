import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from sphere_triangulation import sphere_triangulation
import tensorflow as tf
import gc

def triangles_to_edges(faces):
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

def collect_tfrecords(tfrecords_list, filename):
	#Collects list of tfrecords files into one file
	output = tf.data.TFRecordDataset(tfrecords_list[0])
	for file in tfrecords_list[1:]:
		output = output.concatenate(tf.data.TFRecordDataset(file))
	writer = tf.data.experimental.TFRecordWriter(filename)
	writer.write(output)

def chunk_to_tfrecords(arr, nodes, tfrecords_name, lats, longs):
	#Writes chunk of data from grib into tfrecord

	#Reshaping data for input format
	arr = np.transpose(arr, axes=[1, 3, 4, 2, 0])
	arr = arr.reshape(*arr.shape[:3], -1)

	#Interpolating grid onto graph nodes
	interpolated = []
	for frame in arr:
		interp = RGI((lats[::-1], longs), frame[::-1])
		interpolated.append(interp(nodes))
	
	#Writing into tfrecords
	arr = tf.convert_to_tensor(np.stack(interpolated))
	ds = tf.data.Dataset.from_tensor_slices(arr)
	ds = ds.map(tf.io.serialize_tensor)
	writer = tf.data.experimental.TFRecordWriter(tfrecords_name)
	writer.write(ds)

def grib_to_tfrecords(grib_filename, long_res, lat_res, longs, lats, n_times):
	#Preprocesses grib file into tfrecords
	print('Decoding ' + grib_filename)
	nodes, faces = sphere_triangulation(long_res, lat_res)

	#Opening fileand creating chunks of size around 30
	ds = xr.open_dataset(grib_filename, engine='cfgrib')
	chunks = np.array_split(range(n_times), int(n_times/30))

	#Decoding and writing chunks
	filenames = []
	for i, chunk in enumerate(chunks):
		print('Preprocessing chunk of times: ' + str(chunk))
		#Writing chunk
		arr = ds.isel(time=chunk).to_array().to_numpy()
		filename='part_' + str(i) + '.tfrecords'
		chunk_to_tfrecords(arr, nodes, filename, lats, longs)
		
		#Clean memory
		del arr
		gc.collect()
		print('Created chunk tfrecords file: ' + filename)

		filenames.append(filename)

	collect_tfrecords(filenames, grib_filename + '.tfrecords')
	print('Collected tfrecords: ' + grib_filename + '.tfrecords')

	#Calculating graph structure
	simplices = tf.convert_to_tensor(faces)
	edges = triangles_to_edges(faces)

	u_i = tf.transpose(tf.gather(nodes, edges[0]))
	u_j = tf.transpose(tf.gather(nodes, edges[1]))

	u_j = tf.stack([tf.cos(u_j[0])*tf.cos(u_j[1]), tf.cos(u_j[0])*tf.sin(u_j[1]), tf.sin(u_j[0])])
	u_i = tf.stack([tf.cos(u_i[0])*tf.cos(u_i[1]), tf.cos(u_i[0])*tf.sin(u_i[1]), tf.sin(u_i[0])])

	u_ij = u_i - u_j
	u_ij_norm = tf.expand_dims(tf.norm(u_ij, axis=0), 0)

	edge_features = tf.transpose(tf.concat([u_ij, u_ij_norm], axis=0))

	#Writing graph structure
	edge_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(edges, 0))
	edge_feats_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(edge_features, 0))

	#Converting to tfrecords format
	edge_ds = edge_ds.map(tf.io.serialize_tensor)
	edge_feats_ds = edge_feats_ds.map(tf.io.serialize_tensor)
	writer = tf.data.experimental.TFRecordWriter('edges.tfrecords')
	writer.write(edge_ds)
	writer = tf.data.experimental.TFRecordWriter('edge_features.tfrecords')
	writer.write(edge_feats_ds)
