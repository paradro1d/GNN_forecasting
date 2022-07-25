import tensorflow as tf
from tensorflow.data import AUTOTUNE

def load_tfrecords(n_feats_f_tr, n_feats_f_val, e_feats_f, edges):
	#Reads tfrecords format datasets and decodes them

	#Concatenate all training nodes files
	nodes_ds_tr = tf.data.TFRecordDataset(n_feats_f_tr[0], num_parallel_reads=AUTOTUNE, compression_type='GZIP')
	for f in n_feats_f_tr[1:]:
		app = tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE, compression_type='GZIP')
		nodes_ds_tr = nodes_ds_tr.concatenate(app)
	nodes_ds_tr = nodes_ds_tr.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float64))

	#Concatenate all validation nodes files
	nodes_ds_val = tf.data.TFRecordDataset(n_feats_f_val[0], num_parallel_reads=AUTOTUNE, compression_type='GZIP')
	for f in n_feats_f_tr[1:]:
		app = tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE, compression_type='GZIP')
		nodes_ds_val = nodes_ds_val.concatenate(app)
	nodes_ds_val = nodes_ds_val.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float64))


	#Prepare graph structure data
	edge_feats_ds = tf.data.TFRecordDataset(edge_features_file, num_parallel_reads=AUTOTUNE)
	edge_feats_ds = edge_feats_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float64)).repeat()

	edges_ds = tf.data.TFRecordDataset(edges, num_parallel_reads=AUTOTUNE)
	edges_ds = edges_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.int32)).repeat()

	return nodes_ds_tr, nodes_ds_val, edge_feats_ds, edges_ds

def zip_datasets(nodes_ds, edge_feats_ds, edges_ds, batch_size):
	#Preprocesses dataset for model input format

	#Zipping dataset for 4 predictions ahead
	ds_tup = (nodes_ds, nodes_ds.skip(1), nodes_ds.skip(2), 
		nodes_ds.skip(3), nodes_ds.skip(4))
	nodes_ds = tf.data.Dataset.zip(ds_tup)

	#Filtering data
	filter_nan = lambda a, b, c, d, e: not tf.reduce_any(tf.math.is_nan(a))\
		and not tf.reduce_any(tf.math.is_nan(b))\
		and not tf.reduce_any(tf.math.is_nan(c))\
		and not tf.reduce_any(tf.math.is_nan(d))\
		and not tf.reduce_any(tf.math.is_nan(e))
	nodes_ds = nodes_ds.filter(filter_nan)

	input_ds = tf.data.Dataset.zip((edges_ds, edge_feats_ds,
		 nodes_ds)).batch(batch_size).prefetch(AUTOTUNE)
	return input_ds
