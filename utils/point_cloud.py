import numpy as np
import tensorflow as tf
import argparse
from  gauss_kernel import smoothing_kernel

pc_feat_stop_points_gradient=True
def pc_normalization(pc,numpy=True):
    # pointcloud normalization
    # args:
    #         pc: [N,3]
    #         numpy: Tensor or np
    # return:
    #         normalized pc
    if(numpy==True):
            centroid = np.mean(pc, axis=0)
            pc -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(pc)**2,axis=-1)))
            pc /= furthest_distance
    else:
            pc = tf.cast(pc,tf.float32)
            centroid = tf.reduce_mean(tf.cast(pc,tf.float32),axis=0)
            furthest_distance = tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(tf.abs(pc)),axis=-1)))
            pc = tf.truediv(pc,furthest_distance)
    return pc

def multi_expand(inp, axis, num):
    inp_big = inp
    for i in range(num):
        inp_big = tf.expand_dims(inp_big, axis)
    return inp_big


# def pointcloud2voxels(cfg, input_pc, sigma):  # [B,N,3]
#     # TODO replace with split or tf.unstack
#     x = input_pc[:, :, 0]
#     y = input_pc[:, :, 1]
#     z = input_pc[:, :, 2]

#     vox_size = cfg.vox_size

#     rng = tf.linspace(-1.0, 1.0, vox_size)
#     xg, yg, zg = tf.meshgrid(rng, rng, rng)  # [G,G,G]

#     x_big = multi_expand(x, -1, 3)  # [B,N,1,1,1]
#     y_big = multi_expand(y, -1, 3)  # [B,N,1,1,1]
#     z_big = multi_expand(z, -1, 3)  # [B,N,1,1,1]

#     xg = multi_expand(xg, 0, 2)  # [1,1,G,G,G]
#     yg = multi_expand(yg, 0, 2)  # [1,1,G,G,G]
#     zg = multi_expand(zg, 0, 2)  # [1,1,G,G,G]

#     # squared distance
#     sq_distance = tf.square(x_big - xg) + tf.square(y_big - yg) + tf.square(z_big - zg)

#     # compute gaussian
#     func = tf.exp(-sq_distance / (2.0 * sigma * sigma))  # [B,N,G,G,G]

#     # normalise gaussian
#     if cfg.pc_normalise_gauss:
#         normaliser = tf.reduce_sum(func, [2, 3, 4], keep_dims=True)
#         func /= normaliser
#     elif cfg.pc_normalise_gauss_analytical:
#         # should work with any grid sizes
#         magic_factor = 1.78984352254  # see estimate_gauss_normaliser
#         sigma_normalised = sigma * vox_size
#         normaliser = 1.0 / (magic_factor * tf.pow(sigma_normalised, 3))
#         func *= normaliser

#     summed = tf.reduce_sum(func, axis=1)  # [B,G,G G]
#     voxels = tf.clip_by_value(summed, 0.0, 1.0)
#     voxels = tf.expand_dims(voxels, axis=-1)  # [B,G,G,G,1]

#     return voxels

#NOICE: in pointnet frontend.  Each point cloud contains 2048 points uniformly sampled from a shape surface. 
#       Each cloud is zero-mean and normalized into an unit sphere.
def pointcloud2voxels3d_fast(vox_size, pc, feat):  # [B,N,3]
    """Args:
            cfg: TODO---voxel density input
            pc: point cloud coords with B*N*3
            feat: point cloud features B*N*Feas
        Ret:
            pc density in each voxel
            pc feat values in grids
    """
    # TODO: voxel size should be specified in the input arguments
    
    batch_size = pc.shape[0]
    num_points = tf.shape(pc)[1]
    num_feat = feat.shape[-1]

    # if no feat data is existed
    has_feat = feat is not None

    grid_size = 2.0 # a cubic of x,y,z range in [-1,1]
    half_size = grid_size / 2

    filter_outliers = True
    valid = tf.logical_and(pc >= -half_size, pc <= half_size)
    valid = tf.reduce_all(valid, axis=-1)

    vox_size_tf = tf.constant([[[vox_size, vox_size, vox_size]]], dtype=tf.float32) # v*v*v
    pc_grid = (pc + half_size)/2.0 * (vox_size_tf - 1) # affine to [0,1] then transform to grid idx [0,63]
    indices_floor = tf.floor(pc_grid)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)

    batch_indices = tf.expand_dims(batch_indices, -1) # B*1
    batch_indices = tf.tile(batch_indices, [1, num_points]) # B*N
    batch_indices = tf.expand_dims(batch_indices, -1) # B*N*1



    indices = tf.concat([batch_indices, indices_int], axis=2) # B*N*4
    indices = tf.reshape(indices, [-1, 4]) #[N,4]


    r = pc_grid - indices_floor  # fractional part
    rr = [1.0 - r, r]
    if filter_outliers:
        valid = tf.reshape(valid, [-1])
        indices = tf.boolean_mask(indices, valid)

    vx = tf.zeros([batch_size,vox_size,vox_size,vox_size],dtype=tf.float32)
    vx_feat = tf.zeros([batch_size,vox_size,vox_size,vox_size,num_feat],dtype=tf.float32)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                # vx, vx_feat = interpolate_scatter3d([k, j, i])
                # (1-x_d)*y_d
                updates_raw = rr[k][:, :, 0] * rr[j][:, :, 1] * rr[i][:, :, 2]

                updates = tf.reshape(updates_raw, [-1]) #[B,N,1]
                if filter_outliers:
                    updates = tf.boolean_mask(updates, valid)

                indices_loc = indices
                indices_shift = tf.constant([[0] + [k,j,i]])
                num_updates = tf.shape(indices_loc)[0]
                indices_shift = tf.tile(indices_shift, [num_updates, 1])
                indices_loc = indices_loc + indices_shift

                # 将每一点作为值为1的量，并将其分配到对应的voxel中作为密度函数，最后得到的是
                # 每个voxel的密度
                voxels = tf.scatter_nd(indices_loc, updates, [batch_size, vox_size, vox_size, vox_size])
                if has_feat:
                    if pc_feat_stop_points_gradient:
                        updates_raw = tf.stop_gradient(updates_raw)
                    
                    # voxel densities 只是一个数，而feat需要在每个数下再扩展dimension
                    updates_raw = tf.reshape(updates_raw,[batch_size,num_points,-1])
                    updates_feat = tf.expand_dims(updates_raw, axis=-1) * feat # 

                    updates_feat = tf.reshape(updates_feat, [-1, num_feat])

                    if filter_outliers:
                        updates_feat = tf.boolean_mask(updates_feat, valid)
                    voxels_feat = tf.scatter_nd(indices_loc, updates_feat, [batch_size, vox_size, vox_size, vox_size, num_feat])
                else:
                    voxels_feat = None
                vx = tf.add_n([vx,voxels])
                vx_feat = tf.add_n([vx_feat,voxels_feat]) if has_feat else None

    return vx, vx_feat


def smoothen_voxels3d(voxels, kernel):
    # if cfg.pc_separable_gauss_filter:
    for krnl in kernel:
        voxels = tf.nn.conv3d(voxels, krnl, [1, 1, 1, 1, 1], padding="SAME")
    # else:
    #     voxels = tf.nn.conv3d(voxels, kernel, [1, 1, 1, 1, 1], padding="SAME")
    return voxels

# TODO: make gaussion smoothing layers variable
def convolve_rgb(voxels_feat, kernel):
    # channels becomes column vector
    channels = [voxels_feat[:, :, :, :, k:k+1] for k in range(1088)]
    gaussion_layers = 3
    #three gaussion smoonthing layers
    #for i in range(gaussion_layers):
    for krnl in kernel:
        for i in range(1088):
            vx_feat_c = voxels_feat[:, :, :, :, i:i+1]
            channels[i] = tf.nn.conv3d(channels[i], krnl, [1, 1, 1, 1, 1], padding="SAME")
    out = tf.concat(channels, axis=4)
    return out

def pointcloud_project_fast(vox_size, point_cloud,
                            pts_feat, kernel=None, scaling_factor=None, focal_length=None):
    has_feat = pts_feat is not None

    #TODO: Add camera transformation to here
    voxels, voxels_feat = pointcloud2voxels3d_fast(vox_size, point_cloud, pts_feat)
    voxels = tf.expand_dims(voxels, axis=-1) # [B,V,V,V]--->[B,V,V,V,1]
    voxels_raw = voxels

    voxels = tf.clip_by_value(voxels, 0.0, 1.0)
    pc_rgb_divide_by_occupancies_epsilon = 0.01

    #
    if kernel is not None:
        voxels = smoothen_voxels3d(voxels, kernel)
        if has_feat:
        #     if not cfg.pc_rgb_clip_after_conv:
        #         voxels_rgb = tf.clip_by_value(voxels_rgb, 0.0, 1.0)
            voxels_feat = convolve_rgb(voxels_feat, kernel)

    if scaling_factor is not None:
        sz = scaling_factor.shape[0]
        scaling_factor = tf.reshape(scaling_factor, [sz, 1, 1, 1, 1])
        voxels = voxels * scaling_factor
        voxels = tf.clip_by_value(voxels, 0.0, 1.0)

    if has_feat:
        voxels_div = tf.stop_gradient(voxels_raw)
        voxels_div = smoothen_voxels3d(voxels_div, kernel)
        
        #TODO replace occupancies epsilon with a new arguments from input
        voxels_feat = voxels_feat / (voxels_div + pc_rgb_divide_by_occupancies_epsilon)

        # if cfg.pc_rgb_clip_after_conv:
        #     voxels_rgb = tf.clip_by_value(voxels_rgb, 0.0, 1.0)
    return voxels_feat

def pointcloud_project(gauusion_kernel_size, voxel_size, pc, feats):
    # pc to voxel projection
    # args:
    #     gaussion_kernel_size: size of gaussion smoothing kernel
    #     voxel_size:grid size in xyz
    #     pc: point_cloud coords
    #     feats:point_cloud features
    # return:
    #     porjected features on voxels
    
    # build gaussion smoothing kernel
    kernels = smoothing_kernel(gauusion_kernel_size)

    #pc feat to voxel feature transformation
    voxel_feat = pointcloud_project_fast(voxel_size,pc,feats,kernels)

    voxel_feat = tf.squeeze(voxel_feat,axis=0)

    return voxel_feat

def pointcloud_reverse_project(voxel_feat,voxel_size,coord):
    # reverse trilinear projection
    # args:
    #     voxel_feat: [N,feat_dims]
    #     voxel_size: num of voxels
    #     mesh:       [N,3]
    # return:
    #     feats projected to voxel grids
    
    mesh = pc_normalization(coord,numpy=False)

    #affine transform coord to [0,grid_size]
    mesh = (mesh + 1) / 2  * (voxel_size - 1)

    # feat dims
    dim = tf.shape(voxel_feat)[-1]

    #voxel coordinates of meshes
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
    return tf.concat([coord,output], 1)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_res', type=int, default=32, help='voxel size in x,y,z coordinates [default:64]')
    parser.add_argument('--gauss_kernel_size', type=int, default=11, help='Gaussion smoothing kernel size [default:11]')
    FLAGS = parser.parse_args()

    # virtual pointcloud
    pts = tf.truncated_normal(shape=[1,1024,3],name='pts_coords')
    # virtual features
    feats = tf.truncated_normal(shape=[1,1024,1088],name='pts_feat')

    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #kernels
        kernels = smoothing_kernel(FLAGS)
        voxels_feat = pointcloud_project_fast(FLAGS,pts,feats,kernel=kernels)
        vol,vol_feat = pointcloud2voxels3d_fast(FLAGS,pts,feats)
        print (sess.run(tf.shape(voxels_feat)))

# def pc_point_dropout(points, rgb, keep_prob):
#     shape = points.shape.as_list()
#     num_input_points = shape[1]
#     batch_size = shape[0]
#     num_channels = shape[2]
#     num_output_points = tf.cast(num_input_points * keep_prob, tf.int32)

#     def sampler(num_output_points_np):
#         all_inds = []
#         for k in range(batch_size):
#             ind = np.random.choice(num_input_points, num_output_points_np, replace=False)
#             ind = np.expand_dims(ind, axis=-1)
#             ks = np.ones_like(ind) * k
#             inds = np.concatenate((ks, ind), axis=1)
#             all_inds.append(np.expand_dims(inds, 0))
#         return np.concatenate(tuple(all_inds), 0).astype(np.int64)

#     selected_indices = tf.py_func(sampler, [num_output_points], tf.int64)
#     out_points = tf.gather_nd(points, selected_indices)
#     out_points = tf.reshape(out_points, [batch_size, num_output_points, num_channels])
#     if rgb is not None:
#         num_rgb_channels = rgb.shape.as_list()[2]
#         out_rgb = tf.gather_nd(rgb, selected_indices)
#         out_rgb = tf.reshape(out_rgb, [batch_size, num_output_points, num_rgb_channels])
#     else:
#         out_rgb = None
#     return out_points, out_rgb


# def subsample_points(xyz, num_points):
#     idxs = np.random.choice(xyz.shape[0], num_points)
#     xyz_s = xyz[idxs, :]
#     return xyz_s
