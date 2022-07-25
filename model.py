import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, Add

def build_block(network_structure: dict):
	inp = network_structure['input']
	width = network_structure['width']
	out = network_structure['output']

	input_layer = Input(shape=inp)
	model_layer = Dense(width, activation='relu')(input_layer)
	model_layer = Dense(width, activation='relu')(model_layer)

	model_layer = Dense(out)(model_layer)
	return tf.keras.models.Model(inputs=input_layer, outputs=model_layer)


class Gnn(tf.keras.Model):
	#Basic graph neural network model

	def __init__(self, nn_params):
		#Initialization of model with given network parameters
		super(Gnn, self).__init__()
		self.e_encod = build_block(nn_params['edge_encoder'])
		self.n_encod = build_block(nn_params['node_encoder'])

		self.e_proc = [build_block(nn_params['edge_processor']) for _ in range(nn_params['n_passes'])]
		self.n_proc = [build_block(nn_params['node_processor']) for _ in range(nn_params['n_passes'])]

		self.decod = build_block(nn_params['decoder'])

	def compile(self, optimizer, loss_function, normalizer):
		#Model compilation
		super(Gnn, self).compile()
		self.loss = loss_function
		self.optimizer = optimizer
		self.normalizer = normalizer
		
	def call(self, batch):
		#Model inference on batches
		edges = batch[0]
		n_feats = batch[1]
		e_feats = batch[2]

		n_feats = tf.cast(n_feats, tf.float32)
		e_feats = tf.cast(e_feats, tf.float32)

		n_feats = self.normalizer.normalize_nodes(n_feats)
		e_feats = self.normalizer.normalize_edges(e_feats)

		n_feats = tf.map_fn(lambda x:self.call_unbatched(*x), 
			(edges, n_feats, e_feats))[1]
		return self.normalizer.unnormalize_nodes(n_feats)

	def call_unbatched(self, edges, n_feats, e_feats):
		#Model inference on data in unbatched format
		
		#Encoding features
		n_feats_out = self.n_encod(n_feats)
		e_feats = self.e_encod(edge_features)

		#Processing encoded features
		for n_pr, e_pr in zip(self.node_processors, self.edge_processors):

		#Finding nodes, adjucent to edges and processing them.
			nodes_for_edges = tf.gather(n_feats_out, edges)
			nodes_for_edges = tf.reshape(nodes_for_edges, 
					(tf.shape(nodes_for_edges)[0], -1))
			edge_features = (e_pr(tf.concat([e_feats, nodes_for_edges],
							 axis=1)) + e_feats)*0.5

		#Process nodes by summarizing all adjucent edges encodings
			n = tf.shape(n_feats)[0]
			edges_sum = tf.math.unsorted_segment_sum(e_feats, edges[:, 0], n)
			inputs = tf.concat([n_feats_out, edges_sum], axis=1)
			n_feats_out = (n_pr(inputs) + node_features_out)*0.5
        
		#Decoding
		n_feats_out = self.decoder(n_feats_out)
		n_feats = n_feats + n_feats_out
		return edges, n_feats_out, e_feats

	@tf.function(jit_compile=True)
	def train_step(self, batch):
		#Training step

		#Reading input
		edges = batch[0]
		e_feats = batch[1]
		n_feats = batch[2][0]
		#shape = tf.shape(batch[2][0])

		with tf.GradientTape() as tape:
			#Calculating multi-step loss
			loss = 0
			pred = self.call((edges, n_feats, e_feats))
			pred_norm = self.normalizer.normalize_nodes(pred)

			for step in batch[2][1:-1]:
				step = tf.cast(step, tf.float32)
				step = self.normalizer.normalize_nodes(step)

				loss += self.loss(step, pred_norm)

				pred = self.call((edges, pred, e_feats))
				pred_norm = self.normalizer.normalize_nodes(pred)

			step = batch[2][-1]
			step = tf.cast(step, tf.float32)
			step = self.normalizer.normalize_nodes(step)
			loss += self.loss(step, pred_norm)

		#Calculating and applying gradients
		trainable_variables = sum([i.trainable_variables for i in self.node_processors + self.edge_processors], [])
		trainable_variables += (self.edge_encoder.trainable_variables
			+ self.node_encoder.trainable_variables
			+ self.decoder.trainable_variables)
		grads = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(grads, trainable_variables))
		return {'Loss':loss}

	@tf.function(jit_compile=True)
	def test_step(self, batch):
		#Test step for evaluating performance

		#Reading input
		edges = batch[0]
		e_feats = batch[1]
		n_feats = batch[2][0]

		#Calculating multi-step loss
		loss = 0
		pred = self.call((edges, batch[2][0], edge_features))
		pred_norm = self.normalizer.normalize_nodes(prediction)

		for step in batch[2][1:-1]:
			step = tf.cast(step, tf.float32)
			step = self.normalizer.normalize_nodes(step)

			loss += self.loss(step, prediction_norm)

			prediction = self.call((edges, prediction, edge_features))
			prediction_norm = self.normalizer.normalize_nodes(prediction)

		step = batch[2][-1]
		step = tf.cast(step, tf.float32)
		step = self.normalizer.normalize_nodes(step)
		loss += self.loss(prediction_norm, step)

		return {'Loss':loss}
