import tensorflow as tf
from p2m_chamfer import *

def laplace_coord(pred, params, block_id):
	vertex = tf.concat([pred, tf.zeros([1,3])], 0)# [N+1,3] add 零degree 用于填补后面[:8]中不存在的degree
	indices = params['lape_idx'][block_id-1][:, :8] # 正方形mesh中分割出的triangular mesh，最开始的degree8，pooling之后的新点deg=6
	weights = tf.cast(params['lape_idx'][block_id-1][:,-1], tf.float32)

	weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1,1]), [1,3]) # [1,3]: lapacian weights 1/||N_p||, then apply to 3 coords
	laplace = tf.reduce_sum(tf.gather(vertex, indices), 1) #[N,8,3]
	laplace = tf.subtract(pred, tf.multiply(laplace, weights)) # []
	return laplace

def laplace_loss(pred1, pred2, params, block_id):
	# laplace term
	lap1 = laplace_coord(pred1, params, block_id)
	lap2 = laplace_coord(pred2, params, block_id)
	#laplace loss defined as sum(k_coords/||Nk||)
	laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1,lap2)), 1)) * 1500

	move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
	move_loss = tf.cond(tf.equal(block_id,1), lambda:0., lambda:move_loss) #1th block doesn't count move loss, since it is original input VERSUS first deformation
	#defined as total laplaceloss = p_loss-laplace_loss
	return laplace_loss + move_loss
	
def unit(tensor):
	return tf.nn.l2_normalize(tensor, dim=1)

def mesh_loss(pred, params, labels, block_id):
	gt_pt = labels[:, :3] # gt points #groundtruth_points
	gt_nm = labels[:, 3:] # gt normals #groundtruth_normal

	# edge in graph
	# extracts edge id in different mesh resolution: block_id-1 represents the resolution of mesh, then tf.gather() to get coord of pts
	nod1 = tf.gather(pred, params['edges'][block_id-1][:,0]) # edges is [N,2] int32
	nod2 = tf.gather(pred, params['edges'][block_id-1][:,1])
	edge = tf.subtract(nod1, nod2) #raw vector of length

	# edge length loss regularization, penalize flying-out loss, eliminate outlier
	edge_length = tf.reduce_sum(tf.square(edge), 1) # for each edge (x1-x2)^2 + (y1-y2)^2
	edge_loss = tf.reduce_mean(edge_length) * 300 # mean value edge length * 300

	# chamer distance
	# get the nearest neighbor of each pts at gt_pt in/from pred and compute dist dist1, conversely, compute dist2 for NN in gt_pts
	dist1,idx1,dist2,idx2 = nn_distance(gt_pt, pred)
	point_loss = (tf.reduce_mean(dist1) + 0.55*tf.reduce_mean(dist2)) * 3000

	# normal cosine loss
	normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
	normal = tf.gather(normal, params['edges'][block_id-1][:,0]) #extract gt normal of idx of first nods
	cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1)) # normal(p1)*edge(p1p2), unit=l2 normalization
	# cosine = tf.where(tf.greater(cosine,0.866), tf.zeros_like(cosine), cosine) # truncated
	normal_loss = tf.reduce_mean(cosine) * 0.5

	total_loss = point_loss + edge_loss + normal_loss
	return total_loss
