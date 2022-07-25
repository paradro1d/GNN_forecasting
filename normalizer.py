import tensorflow as tf

class Normalizer():
	#Builds pipeline for normalizing features

	def __init__(self, max_accumulations=10**6, std_epsilon=1e-8):
		self.max_accumulations = max_accumulations
		self.std_epsilon=tf.cast(std_epsilon, tf.float64)

	def get_stats_edges(self, edge_dataset):
		#Obtaining statistic parameters for edges
		self.edge_mean = 0
		self.edge_std = 0
		counter = 0

		for batch in edge_dataset:
			counter += tf.shape(batch)[0]*tf.shape(batch)[1]

			self.edge_mean += tf.reduce_sum(batch, axis=[0, 1])
			self.edge_std += tf.reduce_sum(batch**2, axis=[0, 1])

			if counter >= self.max_accumulations:
				break

		counter = tf.cast(counter, tf.float64)

		self.edge_mean = self.edge_mean/counter
		self.edge_std = self.edge_std/counter

		epsilon = tf.fill(tf.shape(self.edge_std), self.std_epsilon)
		self.edge_std = tf.math.sqrt(self.edge_std - self.edge_mean**2)
		self.edge_std = tf.reduce_max(tf.stack([self.edge_std, epsilon]), axis=0)

		self.edge_mean = tf.cast(self.edge_mean, tf.float32)
		self.edge_std = tf.cast(self.edge_std, tf.float32)


	def get_stats_nodes(self, node_dataset):
		#Obtaining statistic parameters for nodes
		self.node_mean = 0
		self.node_std = 0
		counter = 0

		for batch in node_dataset:
			counter += tf.shape(batch)[0]*tf.shape(batch)[1]

			self.node_mean += tf.reduce_sum(batch, axis=[0, 1])
			self.node_std += tf.reduce_sum(batch**2, axis=[0, 1])

			if counter >= self.max_accumulations:
				break

		counter = tf.cast(counter, tf.float64)

		self.node_mean = self.node_mean/counter
		self.node_std = self.node_std/counter

		epsilon = tf.fill(tf.shape(self.node_std), self.std_epsilon)
		self.node_std = tf.math.sqrt(self.edge_std - self.edge_mean**2)
		self.node_std = tf.reduce_max(tf.stack([self.node_std, epsilon]), axis=0)

		self.node_mean = tf.cast(self.node_mean, tf.float32)
		self.node_std = tf.cast(self.node_std, tf.float32)

	#Normalization functions
	def normalize_edges(self, edge_features):
		return (edge_features - self.edge_mean)/self.edge_std

	def normalize_nodes(self, node_features):
		return (node_features - self.node_mean)/self.node_std

	def unnormalize_edges(self, edge_features):
		return edge_features*self.edge_std + self.edge_mean

	def unnormalize_nodes(self, node_features):
		return node_features*self.node_std + self.node_mean
