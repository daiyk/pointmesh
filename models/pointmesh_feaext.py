import tensorflow as tf
import argparse
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import pickle
from point_cloud import pointcloud_project, pointcloud_reverse_project
from T_transform_net import input_transform_net, feature_transform_net
from p2m_losses import mesh_loss, laplace_loss

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(num_point, 6)) # including points and normals
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, params, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])
    print('net transformed size is {} '.format(net_transformed.shape))

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                            padding='VALID', scope='maxpool')
    
    #tile the second dimension
    net = tf.tile(net,[1,num_point,1,1])
    #concatenate loc and glo features
    net = tf.concat([net_transformed,net],-1)
    #squzzes
    tf.squeeze(net,axis=2) #B*N*1*1088 ---> B*N*1088

    #define placeholder for the processing of data
    voxel_feats = pointcloud_project(params['gaussion_kernel_size'],
                                    params['voxel_size'],point_cloud,net)

    #resnet model
    #1th gcn deformation block
    eltwise = [2,4,6,8,10,12]
    with tf.variable_scope('gcn'):
        gcn_net = pointcloud_reverse_project(voxel_feats, params['voxel_size'],params['mesh'])
        support = params['support1']
        last_layer = gcn_net
        last_last_layer = 0
        for i in range(13):
            last_last_layer = last_layer
            last_layer = gcn_net
            gcn_net = tf_util.graphconv(gcn_net, params['graph_hidden'], support=support,
                                        scope='graphcnvblock1_'+str(i),is_training=True,
                                        bn_decay=bn_decay)
            if i in eltwise:
                gcn_net = tf.add(gcn_net,last_last_layer)*0.5
        gcn_net = tf_util.graphconv(gcn_net, params['coord_dim'], support=support,
                                        scope='graphcnvblock1_'+str(13),is_training=True,
                                        bn_decay=bn_decay)
        gcn_net_output1 = gcn_net
        gcn_net_unpooloutput1 = tf_util.graphpooling(gcn_net_output1,params['pool_idx'][0])
        #end of gcn block 1

        #start of gcn block 2
        gcn_net = pointcloud_reverse_project(voxel_feats, params['voxel_size'],gcn_net)
        gcn_net = tf.concat([gcn_net,last_layer],1)
        support = params['support2']
        #unpool mesh
        gcn_net = tf_util.graphpooling(gcn_net,params['pool_idx'][0])
        last_layer = gcn_net
        for i in range(13):
            last_last_layer = last_layer
            last_layer = gcn_net
            gcn_net = tf_util.graphconv(gcn_net,params['graph_hidden'], support=support,
                                        scope='graphcnvblock2_'+str(i),is_training=True,
                                        bn_decay=bn_decay)
            if i in eltwise:
                gcn_net = tf.add(gcn_net,last_last_layer)*0.5
        gcn_net = tf_util.graphconv(gcn_net,params['coord_dim'], support=support,
                                        scope='graphcnvblock2_'+str(13),is_training=True,
                                        bn_decay=bn_decay)
        gcn_net_output2 = gcn_net
        gcn_net_unpooloutput2 = tf_util.graphpooling(gcn_net_output2,params['pool_idx'][1])
        #end of gcn block 2

        #start of gcn block 3
        gcn_net = pointcloud_reverse_project(voxel_feats, params['voxel_size'],gcn_net)
        gcn_net = tf.concat([gcn_net,last_layer],1)
        support = params['support3']
        #unpool mesh
        gcn_net = tf_util.graphpooling(gcn_net,params['pool_idx'][1])
        last_layer = gcn_net
        for i in range(13):
            last_last_layer = last_layer
            last_layer = gcn_net
            gcn_net = tf_util.graphconv(gcn_net,params['graph_hidden'], support=support,
                                        scope='graphcnvblock3_'+str(i),is_training=True,
                                        bn_decay=bn_decay)
            if i in eltwise:
                gcn_net = tf.add(gcn_net,last_last_layer)*0.5
        gcn_net = tf_util.graphconv(gcn_net,params['coord_dim'], support=support,
                                        scope='graphcnvblock3_'+str(13),is_training=True,
                                        bn_decay=bn_decay)
        gcn_net_output3 = gcn_net
    return [gcn_net_output1,gcn_net_output2,gcn_net_output3, gcn_net_unpooloutput1, gcn_net_unpooloutput2],end_points


def get_loss(outputs, end_points, params, labels, weight_decay=0.0, reg_weight=0.001):
    """ args:
            gcn outputs(5), end points
        return:
            Loss
    """
    output1 = outputs[0]
    output2 = outputs[1]
    output3 = outputs[2]
    output1_2 = outputs[3]
    output2_2 = outputs[4]
    inputs = params['mesh']
    # variable L2 loss
    gcn_loss = mesh_loss(output1, params, labels, 1)
    gcn_loss += mesh_loss(output2, params, labels, 2)
    gcn_loss += mesh_loss(output3, params, labels, 3)
    gcn_loss += .1*laplace_loss(inputs, output1, params, 1)
    gcn_loss += laplace_loss(output1_2, output2, params, 2)
    gcn_loss += laplace_loss(output2_2, output3, params, 3)
    var_L2_loss = 0.0
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print(vars)
    for var in vars:
        var_L2_loss = weight_decay * tf.nn.l2_loss(var)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return var_L2_loss + mat_diff_loss * reg_weight + gcn_loss;

# TODO: test code delete later
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_res', type=int, default=32, help='voxel size in x,y,z coordinates [default:64]')
    parser.add_argument('--gauss_kernel_size', type=int, default=11, help='Gaussion smoothing kernel size [default:11]')
    parser.add_argument('--hidden', type=int, default=256, help='Number of units in hidden layer[default:256].')
    parser.add_argument('--feat_dim', type=int, default=1088, help='Number of units in feature layer[default:1088].')
    parser.add_argument('--coord_dim', type=int,default=3, help='Number of units in output layer[default:3].')
    FLAGS = parser.parse_args()

    flag = {
        'voxel_res': FLAGS.voxel_res,
        'gauss_kernel_size': FLAGS.gauss_kernel_size,
        'hidden': FLAGS.hidden,
        'feat_dim': FLAGS.feat_dim,
        'coord_dim': FLAGS.coord_dim
    }

    # placeholder should move to train file
    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3)), #feature is the coords of ellipsoid mesh
        'dropout': None,
        'learning_rate': 1e-05, # initial learning rate
        'voxel_res': flag['voxel_res'],
        'labels': tf.placeholder(tf.float32, shape=(None, 6)),
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],  #for face loss, not used.
        'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], 
        'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], #for laplace term
        'weight_decay': 5e-6, # weight decay for L2 loss
        'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] #for unpooling
    }

    pkl = pickle.load(open('data/ellipsoid/info_ellipsoid.dat', 'rb'),encoding='latin1')
    feed_dict = construct_feed_dict(pkl, placeholders)
    feed_dict.update({placeholders['labels']:np.random.randn(1024,6)})
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    inputs = tf.zeros((1,1024,3))
    outputs = get_model(inputs, tf.constant(True), flag, placeholders)
    sess.run(tf.global_variables_initializer())
    out = sess.run(outputs,feed_dict=feed_dict)
    print(out[2])