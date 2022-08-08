import tensorflow as tf
import numpy as np
from tensorflow.data import AUTOTUNE

def load_ds(*args):
    '''
    Reads and concatenates datasets sequentially
    '''

    dataset = tf.data.TFRecordDataset(args[0], compression_type='GZIP')

    for filename in args[1:]:
        dataset = dataset.concatenate(tf.data.TFRecordDataset(filename,
            compression_type='GZIP'))

    dataset = dataset.map(lambda x: tf.io.parse_tensor(x, out_type=tf.float32))
    return dataset.prefetch(AUTOTUNE)

def concat_values(*args):
    '''
    Prepares prediction values from several datasets, proposing
    that features are flattened
    '''
    return tf.data.Dataset.zip(args).map(lambda *args:
            tf.concat([*args], axis = -1))

@tf.function(jit_compile=True)
def _interpolate(grid, nodes, res=2):
    #Interpolation onto grid nodes
    nodes = tf.cast(nodes, tf.float32)/np.pi*180/res + tf.constant([90/res, 0])
    lat, lon = tf.unstack(nodes, axis=-1)
    grid = tf.pad(grid, tf.constant([[0, 0], [0, 1], [0, 0]]), mode='SYMMETRIC')
    
    #interpolating along lat axis
    lower_lat = tf.floor(lat)
    higher_lat = tf.floor(lat) + 1.0

    #interpolating along lon axis
    lower_lon = tf.floor(lon)
    higher_lon = tf.floor(lon) + 1.0
    
    #Slicing values
    val_00 = tf.gather_nd(grid, tf.cast(tf.stack([lower_lat, lower_lon], axis=-1), tf.int32))
    val_10 = tf.gather_nd(grid, tf.cast(tf.stack([higher_lat, lower_lon], axis=-1), tf.int32))
    val_01 = tf.gather_nd(grid, tf.cast(tf.stack([lower_lat, higher_lon], axis=-1), tf.int32))
    val_11 = tf.gather_nd(grid, tf.cast(tf.stack([higher_lat, higher_lon], axis=-1), tf.int32))\
    
    #interpolating
    values = tf.expand_dims((lat - lower_lat)*(lon - lower_lon), -1)*val_00\
        + tf.expand_dims((higher_lat - lat)*(lon - lower_lon), -1)*val_10\
        + tf.expand_dims((lat - lower_lat)*(higher_lon - lon), -1)*val_01\
        + tf.expand_dims((higher_lat - lat)*(higher_lon - lon), -1)*val_11
    return values

def stat_mesh_ds(predict_dataset, ocean_mask, mesh, res=2, nsteps=4):
    '''
    Prepares initial data by interpolating mesh values onto grid with given res
    and returning dataset in necessary output format for training
    '''
    predict_dataset = predict_dataset.map(lambda x: _interpolate(x,
        mesh['nodes_coords'], res=res), 
        num_parallel_calls=AUTOTUNE,
        deterministic=True)
    ocean_mask = _interpolate(tf.expand_dims(ocean_mask, -1), mesh['nodes_coords'])
    ocean_mask = tf.concat([ocean_mask, tf.cos(mesh['nodes_coords'])], axis=-1)
    const_dataset = tf.data.Dataset.from_tensor_slices(tf.expand_dims(ocean_mask, 0))
    sequential_data = tuple(predict_dataset.skip(i) for i  in range(nsteps))
    sequential_data = tf.data.Dataset.zip(sequential_data)
    edges_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(mesh['edges'], 0))
    edge_features = tf.gather(mesh['nodes_coords'], mesh['edges'])
    x = tf.cos(edge_features[:, :, 0])*tf.cos(edge_features[:, :, 1])
    y = tf.cos(edge_features[:, :, 0])*tf.sin(edge_features[:, :, 1])
    z = tf.sin(edge_features[:, :, 0])
    cart = tf.stack([x, y, z], axis=-1)
    edge_features = cart[:, 0] - cart[:, 1]
    edge_features = tf.concat([tf.expand_dims(tf.linalg.norm(edge_features, axis=-1), 
                                              -1), edge_features], axis=-1)
    edge_feats_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(edge_features,
                                                                      axis=0))
    return tf.data.Dataset.zip((edges_ds.repeat(), 
                                edge_feats_ds.repeat(), 
                                sequential_data, 
                                const_dataset.repeat())).prefetch(AUTOTUNE).batch(1)
