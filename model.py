import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU

def build_net(network_structure):
	#Build network with given structure in list format:
	#	[input, layer1, layer2, ..., output]
	input_layer = keras.layers.Input(shape=(network_structure[0]))
	model_layer = Dense(network_structure[1])(input_layer)
	#model_layer = BatchNormalization()(model_layer)
	model_layer = ReLU()(model_layer)
	for layer in network_structure[2:-1]:
		model_layer = Dense(layer)(model_layer)
		#model_layer = BatchNormalization()(model_layer)
		model_layer = ReLU()(model_layer)
	model_layer = Dense(network_structure[-1])(model_layer)
	return tf.keras.models.Model(inputs=input_layer, outputs=model_layer)


class Gnn_basic(tf.keras.Model):
	#Basic graph neural network model
	
	def __init__(self, edge_encoder, node_encoder, edge_processor, node_processor, decoder, loss_function):
		#Initialization of model with given network parameters
		super(Gnn_basic, self).__init__()
		self.edge_encoder = build_net(edge_encoder)
		self.node_encoder = build_net(node_encoder)
		self.edge_processor = build_net(edge_processor)
		self.node_processor = build_net(node_processor)
		self.decoder = build_net(decoder)
		self.loss_function = loss_function
		
	def compile(self, optimizer):
		#Model compilation
		super(Gnn_basic, self).compile()
		self.optimizer = optimizer
		
	@tf.function()
	def call(self, edges, node_features, edge_features):
		#Model inference on batches
		node_features = tf.cast(node_features, tf.float32)
		edge_features = tf.cast(edge_features, tf.float32)
		return tf.map_fn(lambda x:self.call_unbatched(*x), 
			(edges, node_features, edge_features))[1]

	def call_unbatched(self, edges, node_features, edge_features):
		#Model inference on data in unbatched format
		
		#Encoding feature data
		node_features_out = self.node_encoder(node_features)
		edge_features = self.edge_encoder(edge_features)
		
		#Finding nodes, adjucent to edges and process them(edges).
		nodes_for_edges = tf.transpose(tf.gather(node_features_out, 
							edges), perm=[1, 0, 2])
		nodes_for_edges = tf.reshape(nodes_for_edges, 
					(tf.shape(nodes_for_edges)[0], -1))
		edge_features = edge_processor(tf.concat([edge_features, 
							nodes_for_edges], axis=1))
		
		#Process nodes by summarizing all adjucent edges encodings
		n = tf.shape(node_features)[0]
		node_features_out = self.node_processor(tf.concat([node_features_out, 
			tf.math.unsorted_segment_sum(edge_features, 
				tf.gather(edges, 0), n)], axis=1))
		#Decoding
		node_features_out = self.decoder(node_features_out)
		node_features_out = node_features + node_features_out
		return edges, node_features_out, edge_features

	@tf.function()
	def train_step(self, batch):
		#Training step
		edges = batch[0]
		edge_features = batch[1]
		node_features = batch[2]
		node_features_true = batch[4]
		with tf.GradientTape() as tape:
			node_features_predicted = self.call(edges, node_features, batch)
			loss = self.loss_features(node_features_predicted, node_features_true)
		trainable_variables = tf.concat([self.edge_encoder.trainable_variables, 
			self.edge_processor.trainable_variables,
			self.node_encoder.trainable_variables,
			self.node_processor.trainable_variables,
			self.decoder.trainable_variables])
		grads = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(grads, trainable_variables))
		return {'Loss':loss}
