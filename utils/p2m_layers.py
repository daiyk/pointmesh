
import tensorflow as tf
import tf_util
from data_prep_util import pc_normalization
flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

# img_features=[feat_dim,feature_dim,channels], [112,112,32]......
# change input to trilinear interpolate the value
def project(voxel_feat, mesh, dim):
    '''args:
            voxel_feat: [N,feat_dims]
            mesh:       [N,3]
            dim:        feat_dims
     return:
            feats projected to voxel grids
    '''

    #coordinates of meshes
    X = mesh[:,0] #[N,1]
    Y = mesh[:,1] #[N,1]
    Z = mesh[:,2] #[N,1]

    #pts voxel boundary idx
    x1 = tf.floor(X)
    x2 = tf.ceil(X)
    y1 = tf.floor(Y)
    y2 = tf.ceil(Y)
    z1 = tf.floor(Z)
    z2 = tf.ceil(Z)

    
    N = mesh.shape[0] # Mesh points
    

    # stack,dim=1 在横向，列向量方向叠加
    # gather_nd 由高到低定义抽取的dim，比如[1,2,3],则[x,y] indice抽取前两个[x,y]维度的第三个维度的值
    # voxel_feat has dims [G,G,G,FEAT]
    Q000 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y1,tf.int32),tf.cast(z1,tf.int32)],1)) # [N,feat] N:  mesh numbers. to take #[N,dim],N:pts_num,dim:voxel_feat_dim
    Q001 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y1,tf.int32),tf.cast(z2,tf.int32)],1)) # [N,feat]
    Q010 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y2,tf.int32),tf.cast(z1,tf.int32)],1)) # [N,feat]
    Q011 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y2,tf.int32),tf.cast(z2,tf.int32)],1)) # [N,feat]
    Q100 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y1,tf.int32),tf.cast(z1,tf.int32)],1)) # [N,feat]
    Q101 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y1,tf.int32),tf.cast(z2,tf.int32)],1)) # [N,feat]
    Q110 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y2,tf.int32),tf.cast(z1,tf.int32)],1)) # [N,feat]
    Q111 = tf.gather_nd(voxel_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y2,tf.int32),tf.cast(z2,tf.int32)],1)) # [N,feat]

    # compute weights
    weightx = tf.truediv(tf.subtract(X,x1),tf.subtract(x2,x1)) # [N,]
    weighty = tf.truediv(tf.subtract(Y,y1),tf.subtract(y2,y1)) # [N,]
    weightz = tf.truediv(tf.subtract(Z,z1),tf.subtract(z2,z1)) # [N,]

    # tile weights
    weightx = tf.tile(tf.reshape(weightx,[-1,1]),[1,dim])
    weighty = tf.tile(tf.reshape(weighty,[-1,1]),[1,dim])
    weightz = tf.tile(tf.reshape(weightz,[-1,1]),[1,dim])

    #1th interpolate along x
    Q00 = tf.multiply(Q000,tf.subtract(1.0, weightx)) + tf.multiply(Q100,weightx)
    Q01 = tf.multiply(Q001,tf.subtract(1.0, weightx)) + tf.multiply(Q101,weightx)
    Q10 = tf.multiply(Q010,tf.subtract(1.0, weightx)) + tf.multiply(Q110,weightx)
    Q11 = tf.multiply(Q011,tf.subtract(1.0, weightx)) + tf.multiply(Q111,weightx)

    #2nd interpolate along y
    Q0 = tf.multiply(Q00,tf.subtract(1.0,weighty)) + tf.multiply(Q10,weighty)
    Q1 = tf.multiply(Q01,tf.subtract(1.0,weighty)) + tf.multiply(Q11,weighty)

    #3rd interpolate along z
    output = tf.multiply(Q0,tf.subtract(1.0,weightz)) + tf.multiply(Q1,weightz) # [N, feat]
    return output

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in list(kwargs.keys()):
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=True, gcn_block_id=1,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        # need to define dropout
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if gcn_block_id == 1:
            self.support = placeholders['support1']
        elif gcn_block_id == 2:
            self.support = placeholders['support2']
        elif gcn_block_id == 3:
            self.support = placeholders['support3']
            
        self.sparse_inputs = sparse_inputs
        self.num_output = output_dim
        self.featureless = featureless
        self.bias = bias


        #input_dim=963 is the total number of features,output_dim=256 is the hidden layers
        # helper variable for sparse dropout
        self.num_features_nonzero = 3#placeholders['num_features_nonzero']

        # with tf.variable_scope(self.name + '_vars'):
        #     #graph convolution defines 
        #     for i in range(len(self.support)): # compute graph convolution, the dot is tf.sparse_tensor_dense_dot
        #         # self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
        #         #                                         name='weights_' + str(i))
        #         # use tensorflow glorot initializer
        #         self.vars['weights_'+str(i)] = tf.get_variable(name='weights_'+str(i),shape=[input_dim,output_dim],initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
        #     if self.bias:
        #         self.vars['bias'] = _variable_on_cpu(name='bias',shape=[output_dim],initializer=tf.zeros_initializer())

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        #now dropout according to the layer input variable
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # support 是周边nodes的信息，维度为[N,N]，包括两个矩阵，一个为diagnoal的单位矩阵，用于乘以自己
        # 另一个[N，N]存储了每个点的周边node的weights，用于GNN里面乘它相邻的nodes
        supports = list()
        for i in range(len(self.support)):
            if self.featureless:
                x = tf.ones(shape = tf.shape(inputs))
                # pre_sup = dot(x, self.vars['weights_' + str(i)],
                #               sparse=self.sparse_inputs)
                # use tf_util 2d convolution
            pre_sup = tf_util.fully_connected(x, self.num_output, scope=self.name+str(i),
                         use_xavier=True, activation_fn=None, is_training=True, bn_decay = None) # ???use two bias variables
            support = dot(self.support[i], pre_sup, sparse=True) # 应该是weights1*support(0)*x + weight2*support(1)*x
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        # if self.bias:
        #     output += self.vars['bias']

        return self.act(output)

class GraphPooling(Layer):
    """Graph Pooling layer."""
    def __init__(self, placeholders, pool_id=1, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)

        self.pool_idx = placeholders['pool_idx'][pool_id-1]

    def _call(self, inputs):
        X = inputs

        add_feat = (1/2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
        outputs = tf.concat([X, add_feat], 0)

        return outputs

class GraphProjection(Layer):
    """Graph Pooling layer."""
    def __init__(self, placeholders, **kwargs):
        super(GraphProjection, self).__init__(**kwargs)

        self.voxel_feat = placeholders['voxel_feat']#[Num_voxel_feat,feats]
        self.voxel_size = placeholders['voxel_res'] # voxel resolution / voxel size
    '''
    def _call(self, inputs):
        coord = inputs
        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        #h = (-Y)/(-Z)*248 + 224/2.0 - 1
        #w = X/(-Z)*248 + 224/2.0 - 1 [28,14,7,4]
        h = 248.0 * tf.divide(-Y, -Z) + 112.0
        w = 248.0 * tf.divide(X, -Z) + 112.0

        h = tf.minimum(tf.maximum(h, 0), 223)
        w = tf.minimum(tf.maximum(w, 0), 223)
        indeces = tf.stack([h,w], 1)

        idx = tf.cast(indeces/(224.0/56.0), tf.int32)
        out1 = tf.gather_nd(self.img_feat[0], idx)
        idx = tf.cast(indeces/(224.0/28.0), tf.int32)
        out2 = tf.gather_nd(self.img_feat[1], idx)
        idx = tf.cast(indeces/(224.0/14.0), tf.int32)
        out3 = tf.gather_nd(self.img_feat[2], idx)
        idx = tf.cast(indeces/(224.0/7.00), tf.int32)
        out4 = tf.gather_nd(self.img_feat[3], idx)

        outputs = tf.concat([coord,out1,out2,out3,out4], 1)
        return outputs
    '''
    # input is placeholder['features'] mesh points
    # don't need batch size to participate
    def _call(self, inputs):
        coord = inputs #[N,3]
        # X = inputs[:, 0]
        # Y = inputs[:, 1]
        # Z = inputs[:, 2]
        print('')
        
        # normalization mesh:[N,3] to mean zero and var 1 since the shape is not standard
        coord = pc_normalization(coord,numpy=False)

        # affine transform to [0,grid_size]
        coord = (coord + 1) / 2  * (self.voxel_size - 1)

        # feat_dims
        dims = tf.shape(self.voxel_feat)[-1]

        #trilinear interpolation
        output = project(self.voxel_feat, coord, dims)

        # concatenate with mesh coords

        # # define 
        # h = 250 * tf.divide(-Y, -Z) + 112
        # w = 250 * tf.divide(X, -Z) + 112

        # h = tf.minimum(tf.maximum(h, 0), 223)
        # w = tf.minimum(tf.maximum(w, 0), 223)

        # x = h/(224.0/56)
        # y = w/(224.0/56)
        # out1 = project(self.voxel_feat[0], x, y, 64) #[N,64],N:num_pts

        # x = h/(224.0/28)
        # y = w/(224.0/28)
        # out2 = project(self.voxel_feat[1], x, y, 128) #[N,128],N:num_pts

        # x = h/(224.0/14)
        # y = w/(224.0/14)
        # out3 = project(self.voxel_feat[2], x, y, 256) #[N,256],N:num_pts

        # x = h/(224.0/7)
        # y = w/(224.0/7)
        # out4 = project(self.voxel_feat[3], x, y, 512) #[N,512],N:num_pts
        outputs = tf.concat([inputs,output], 1) # [N,coords+feat] # [N,3+64+128+256+512]
        return outputs


if __name__ =="__main__":
    #test project func
    N = tf.constant(1024,tf.int32)
    feat_dim = tf.constant(512,tf.int32)

    #generate mesh points
    mesh_pts = tf.random_uniform([N,3],maxval=1,dtype=tf.float32)
    mesh_pts = mesh_pts * (64-1)

    #voxel grids
    voxel_feat = tf.random_normal([64,64,64,feat_dim],dtype=tf.float32)

     # test standardlization
    testVal = tf.random_uniform(shape=[1024,3],maxval=1.0,dtype=tf.float32)

    testVal_stand = standardlize_data(testVal)
    mean_stand=tf.reduce_mean(testVal_stand,axis=0)
    testVal2 = tf.nn.moments(testVal_stand,axes=0)

    #test prpject functions
    projected_feats = project(voxel_feat,mesh_pts,feat_dim)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pts = sess.run(mesh_pts)
        print(pts)
        feats = sess.run(projected_feats)
        print(sess.run(testVal2))

   
