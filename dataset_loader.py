import tensorflow as tf
from tensorflow.data import AUTOTUNE

def load_tfrecords(nodes_features_file, edge_features_file, edges):
	nodes_ds = tf.data.TFRecordDataset(nodes_features_file, num_parallel_reads=AUTOTUNE)
	nodes_ds = nodes_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float64)).batch(1)
	edge_feats_ds = tf.data.TFRecordDataset(edge_features_file, num_parallel_reads=AUTOTUNE)
	edge_feats_ds = edge_feats_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float64)).batch(1).repeat()
	edges_ds = tf.data.TFRecordDataset(edges, num_parallel_reads=AUTOTUNE)
	edges_ds = edges_ds.map(lambda x: tf.io.parse_tensor(x, out_type=tf.int32)).batch(1).repeat()
	return nodes_ds, edge_feats_ds, edges_ds
