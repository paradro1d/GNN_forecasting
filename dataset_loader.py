import tensorflow as tf
from tensorflow.data import AUTOTUNE

def load_tfrecords(nodes_features_file, edge_features_file, edges):
	#Reads tfrecords format datasets and decodes them
	nodes_ds = tf.data.TFRecordDataset(nodes_features_file, num_parallel_reads=AUTOTUNE)
	nodes_ds = nodes_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float64))

	edge_feats_ds = tf.data.TFRecordDataset(edge_features_file, num_parallel_reads=AUTOTUNE)
	edge_feats_ds = edge_feats_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float64)).repeat()

	edges_ds = tf.data.TFRecordDataset(edges, num_parallel_reads=AUTOTUNE)
	edges_ds = edges_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.int32)).repeat()

	return nodes_ds, edge_feats_ds, edges_ds

def zip_datasets(nodes_ds, edge_feats_ds, edges_ds):
	#Preprocesses dataset for model input format
	ds_tup = (nodes_ds, nodes_ds.skip(1), nodes_ds.skip(2), 
		nodes_ds.skip(3), nodes_ds.skip(4))
	nodes_ds = tf.data.Dataset.zip(ds_tup)
	filter_nan = lambda a, b, c, d, e: not tf.reduce_any(tf.math.is_nan(a))\
		and not tf.reduce_any(tf.math.is_nan(b))\
		and not tf.reduce_any(tf.math.is_nan(c))\
		and not tf.reduce_any(tf.math.is_nan(d))\
		and not tf.reduce_any(tf.math.is_nan(e))
	nodes_ds = nodes_ds.filter(filter_nan)
	input_ds = tf.data.Dataset.zip((edges_ds, edge_feats_ds,
		 nodes_ds)).batch(1).prefetch(AUTOTUNE)
	return input_ds
