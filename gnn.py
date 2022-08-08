# %% [code]
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, Add
import cloudpickle

DEFAULT_NET = {'width': 256, 'output': 128}
DEFAULT_GNN = {'edge_encoder': DEFAULT_NET,
        'node_encoder': DEFAULT_NET,
        'edge_processor': DEFAULT_NET,
        'node_processor': DEFAULT_NET,
        'decoder': {'width': 256, 'output': 28},
              'n_passes': 2}

def _build_block(network_structure: dict):
    #Building network for model
    width = network_structure['width']
    out = network_structure['output']
    model = tf.keras.Sequential(
        [
            Dense(width, activation='relu'),
            Dense(width, activation='relu'),
            Dense(out)
        ]
    )
    return model


class Gnn(tf.keras.Model):
    '''Basic class for graph neural network. Performs nodes encoding,
    processing and decoding. Each network is a 2-layer MLP initialized by 
    from_dict() method by dictionary in next format:
        {"width": <hidden layer size>, "output": <output size>}
    GNN model is initialized by next dictionary:
        {'edge_encoder': <edge encoder MLP>,
        'node_encoder': <node encoder MLP>,
        'edge_processor': <edge processor MLP>,
        'node_processor': <node processor MLP>,
        'decoder': <decoder MLP>,
        }
    Or can be loaded by load() method.
    '''
    def __init__(self):
        #Initialization of model with given network parameters
        super(Gnn, self).__init__()
        self.from_dict(DEFAULT_GNN)

    def from_dict(self, nn_params: dict):
        #Creating model structure, defined by dictionary
        self.e_encod = _build_block(nn_params['edge_encoder'])
        self.n_encod = _build_block(nn_params['node_encoder'])

        self.e_proc = [_build_block(
            nn_params['edge_processor']) for _ in range(nn_params['n_passes'])]
        self.n_proc = [_build_block(
            nn_params['node_processor']) for _ in range(nn_params['n_passes'])]

        self.decod = _build_block(nn_params['decoder'])

    def load(self, path: str, n: int):
        #Reading model from serialized pkl format

        self.decod = tf.keras.models.load_model(path + '/decod')

        self.e_encod = tf.keras.models.load_model(path + '/e_encod')
        self.n_encod = tf.keras.models.load_model(path + '/n_encod')
        self.n_proc = []
        self.e_proc = []
        for i in range(n):
            self.n_proc.append(tf.keras.models.load_model(path + '/n_proc' + str(i)))
            self.e_proc.append(tf.keras.models.load_model(path + '/e_proc' + str(i)))

    def compile(self, optimizer, loss_function, normalizer):
        #Model compilation with given loss, optimizer and normalizer
        super(Gnn, self).compile()
        self.loss = loss_function
        self.optimizer = optimizer
        self.normalizer = normalizer

    @tf.function(jit_compile=True)
    def call(self, batch: tuple) -> tf.Tensor:
        #Model inference on batches
        edges = batch[0]
        n_feats = batch[1]
        e_feats = batch[2]
        n_feats_const = batch[3]

        n_feats = self.normalizer.normalize_nodes(n_feats)
        e_feats = self.normalizer.normalize_edges(e_feats)
        n_feats = tf.concat([n_feats, n_feats_const], axis=-1)

        n_feats = tf.map_fn(lambda x:self._call_unbatched(*x), 
                (edges, n_feats, e_feats))[1]
        return self.normalizer.unnormalize_nodes(n_feats)

    def _call_unbatched(self, edges: tf.Tensor, n_feats: tf.Tensor, e_feats: tf.Tensor) -> tuple:
    #Model inference on data in unbatched format

        #Encoding features
        n_feats_out = self.n_encod(n_feats)
        e_feats = self.e_encod(e_feats)

        #Processing encoded features
        for n_pr, e_pr in zip(self.n_proc, self.e_proc):

            #Finding nodes, adjucent to edges and processing them.
            nodes_for_edges = tf.gather(n_feats_out, edges)
            nodes_for_edges = tf.reshape(nodes_for_edges, 
                    (tf.shape(nodes_for_edges)[0], -1))
            e_feats = (e_pr(tf.concat([e_feats, nodes_for_edges],
                axis=1)) + e_feats)*0.5

            #Process nodes by summarizing all adjucent edges encodings
            n = tf.shape(n_feats)[0]
            edges_sum = tf.math.unsorted_segment_sum(e_feats, edges[:, 0], n)
            inputs = tf.concat([n_feats_out, edges_sum], axis=1)
            n_feats_out = (n_pr(inputs) + n_feats_out)*0.5
        
        #Decoding
        n_feats_out = self.decod(n_feats_out)
        return edges, n_feats_out, e_feats

    def save(self, path: str):
        model_dict = {
                'optimizer': self.optimizer,
                'normalizer': self.normalizer,
                'loss': self.loss
        }
        self.e_encod.save(path + '/e_encod')
        self.n_encod.save(path +'/n_encod')
        self.decod.save(path + '/decod')
        for i, (e_pr, n_pr) in enumerate(zip(self.e_proc, self.n_proc)):
            e_pr.save(path + '/e_proc' + str(i))
            n_pr.save(path + '/n_proc' + str(i))
        with open(path + '/additions', 'wb') as f:
            cloudpickle.dump(model_dict, f)


class Forecaster(Gnn):
    '''
    Class for forecasting gnn. Makes prediction on a static graph.
    Takes data for training in the next format:
    (<edges>, <edge features>, (<consequent states for prediction>), <const node feats>)
    '''
    @tf.function(jit_compile=True)
    def train_step(self, batch: tuple):
    #Training step

    #Reading input
        edges = batch[0]
        e_feats = batch[1]
        n_feats = batch[2][0]
        const_feats = batch[3]

        with tf.GradientTape() as tape:
        #Calculating multi-step loss
            losses = []
            pred = super().call((edges, n_feats, e_feats, const_feats))
            pred_norm = self.normalizer.normalize_nodes(pred)

            for step in batch[2][1:-1]:
                step = self.normalizer.normalize_nodes(step)

                losses.append(self.loss(step, pred_norm))

                pred = super().call((edges, pred, e_feats, const_feats))
                pred_norm = self.normalizer.normalize_nodes(pred)

            step = batch[2][-1]
            step = self.normalizer.normalize_nodes(step)
            losses.append(self.loss(step, pred_norm))
            loss = sum(losses)

        #Training logs
        loss_dict = {str(i) : loss for i, loss in enumerate(losses)}
        loss_dict.update({'loss':loss})

        #Calculating and applying gradients
        trainable_variables = sum([i.trainable_variables for i in self.n_proc 
                + self.e_proc], [])
        trainable_variables += (self.e_encod.trainable_variables
            + self.n_encod.trainable_variables
            + self.decod.trainable_variables)

        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss_dict

    @tf.function(jit_compile=True)
    def test_step(self, batch: tuple):
        #Test step for evaluating performance

        #Reading input
        edges = batch[0]
        e_feats = batch[1]
        n_feats = batch[2][0]
        const_feats = batch[3]
        losses = []

        #Calculating multi-step loss
        loss = 0
        pred = super().call((edges, n_feats, e_feats, const_feats))
        pred_norm = self.normalizer.normalize_nodes(pred)

        for step in batch[2][1:-1]:
            step = tf.cast(step, tf.float32)
            step = self.normalizer.normalize_nodes(step)

            losses.append(self.loss(step, pred_norm))

            pred = super().call((edges, pred, e_feats, const_feats))
            pred_norm = self.normalizer.normalize_nodes(pred)

        step = batch[2][-1]
        step = self.normalizer.normalize_nodes(step)
        losses.append(self.loss(step, pred_norm))
        loss = sum(losses)

        #Evaluating logs
        loss_dict = {str(i) : loss for i, loss in enumerate(losses)}
        loss_dict.update({'loss':loss})

        return loss_dict


############################################################################################
def Interpolator(Gnn):
    '''
    Class for training a neural network for interpolation.
    Takes data for training in the next format:
    (<edges>, <edge features>, (<corrupted data>, <ground truth>))
    '''
    @tf.function()
    def train_step(self, batch):
    #Training step

    #Reading input
        edges = batch[0]
        e_feats = batch[1]
        n_feats = batch[2][0]
        n_true = batch[2][1]

        with tf.GradientTape() as tape:
            pred = self.call((edges, n_feats, e_feats))
            pred_norm = self.normalizer.normalize_nodes(pred)
            n_true = self.normalizer.normalize_nodes(n_true)
            loss = self.loss(pred_norm, n_true)

        #Calculating and applying gradients
        trainable_variables = sum([i.trainable_variables for i in self.n_proc + self.e_proc], [])
        trainable_variables += (self.e_encod.trainable_variables
            + self.n_encod.trainable_variables
            + self.decod.trainable_variables)

        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return {'Loss': loss}

    @tf.function()
    def test_step(self, batch):
    #Test step for evaluating performance

        #Reading input
        edges = batch[0]
        e_feats = batch[1]
        n_feats = batch[2][0]
        n_true = batch[2][1]

        pred = self.call((edges, n_feats, e_feats))
        pred_norm = self.normalizer.normalize_nodes(pred)
        n_true = self.normalizer.normalize_nodes(n_true)
        loss = self.loss(pred_norm, n_true)

        return {'Loss': loss}

'''
class Gradient_estimator(Interpolator):
    Class for estimating gradients. Takes data for training in next format:
    (<edges>, <edge_features>, (<features map>, <gradient map>)
    Performs Metropolis-Hastings algorithm on a grid.
'''
