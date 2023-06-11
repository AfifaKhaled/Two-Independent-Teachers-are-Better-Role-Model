from .basic_ops3 import *

def multihead_attention_3d(inputs, total_key_filters, total_value_filters,
							output_filters, num_heads, training, output_shape, layer_type='SAME',
							name=None):
	if total_key_filters % num_heads != 0:
		raise ValueError("Key depth (%d) must be divisible by the number of "
						"attention heads (%d)." % (total_key_filters, num_heads))
	if total_value_filters % num_heads != 0:
		raise ValueError("Value depth (%d) must be divisible by the number of "
						"attention heads (%d)." % (total_value_filters, num_heads))
	if layer_type not in ['SAME', 'DOWN', 'UP']:
		raise ValueError("Layer type (%s) must be one of SAME, "
						"DOWN, UP." % (layer_type))

	with tf.variable_scope(
			name,
			default_name="multihead_attention_3d",
			values=[inputs]):

		# produce q, k, v
		q, k, v = compute_qkv_3d(inputs, total_key_filters,
						total_value_filters, layer_type,output_shape)

		# after splitting, shape is [batch, heads, d, h, w, channels / heads]
		q = split_heads_3d(q, num_heads)
		k = split_heads_3d(k, num_heads)
		v = split_heads_3d(v, num_heads)

		# normalize
		key_filters_per_head = total_key_filters // num_heads
		q *= key_filters_per_head**-0.5

		# attention
		x = global_attention_3d(q, k, v, training)
		#x=PTM(x,q,k,v)
		x = combine_heads_3d(x)
		x = Conv3D(x, output_filters, 1, 1, use_bias=True)
		# print('x=',x)
		return x


def PTM(x,query_conv,key_conv,value_conv):
	gamma = tf.Variable(tf.zeros(1))
	print('x=', x)
	m_batchsize, height, width, C = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
	print('x=', x.shape)
	proj_query = query_conv(x).view(m_batchsize, -1, width * height).permute(0, 3, 2, 1)
	print('proj_query=', proj_query)
	proj_key = key_conv(x).view(m_batchsize, -1, width * height)
	print('proj_key=', proj_key)
	energy = tf.matmul(proj_query, proj_key)
	attention = tf.softmax(energy, dim=-1)
	proj_value = value_conv(x).view(m_batchsize, -1, width * height)
	print('proj_value=', proj_value)
	out = tf.matmul(proj_value, attention.permute(0, 3, 2, 1))
	out = out.view(m_batchsize, height, width, C)
	print('out=', out)
	out = gamma * out + x
	print('out=', out)
	return out

def compute_qkv_3d(inputs, total_key_filters, total_value_filters, layer_type,output_shape):
	if layer_type == 'SAME':
		q = Conv3D(inputs, total_key_filters, 1, 1,name="Conv3D_attqs", use_bias=True)
	elif layer_type == 'DOWN':
		q = Conv3D(inputs, total_key_filters, 3, 2,name="Conv3D_attqd", use_bias=True)
	elif layer_type == 'UP':
		q = Deconv3D(inputs, total_key_filters, 3, 2,out_shape=output_shape, name="Conv3D_attqu",use_bias=True)

	# linear transformation for k
	k = Conv3D(inputs, total_key_filters, 1, 1, name="Conv3D_attk",use_bias=True)

	# linear transformation for k
	v = Conv3D(inputs, total_value_filters, 1, 1, name="Conv3D_attv",use_bias=True)
	# print('q=',q)
	# print('k=',k)
	# print('v=',v)
	return q, k, v


def split_heads_3d(x, num_heads):


	return tf.transpose(split_last_dimension(x, num_heads), [0, 4, 1, 2, 3, 5])


def split_last_dimension(x, n):

	old_shape = x.get_shape().dims
	last = old_shape[-1]
	new_shape = old_shape[:-1] + [n] + [last // n if last else None]

	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
	ret.set_shape(new_shape)

	return ret


def global_attention_3d(q, k, v, training, name=None):

	with tf.variable_scope(
			name,
			default_name="global_attention_3d",
			values=[q, k, v]):

		new_shape = tf.concat([tf.shape(q)[0:-1], [v.shape[-1].value]], 0)

		# flatten q,k,v
		q_new = flatten_3d(q)
		k_new = flatten_3d(k)
		v_new = flatten_3d(v)

		# attention
		output = dot_product_attention(q_new, k_new, v_new, bias=None,
					training=training, dropout_rate=0.5, name="global_3d")

		# putting the representations back in the right place
		output = scatter_3d(output, new_shape)

		return output


def reshape_range(tensor, i, j, shape):
	"""Reshapes a tensor between dimensions i and j."""

	target_shape = tf.concat(
			[tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
			axis=0)

	return tf.reshape(tensor, target_shape)


def flatten_3d(x):
	"""flatten x."""

	x_shape = tf.shape(x)
	# [batch, heads, length, channels], length = d*h*w
	x = reshape_range(x, 2, 5, [tf.reduce_prod(x_shape[2:5])])

	return x


def scatter_3d(x, shape):
	"""scatter x."""

	x = tf.reshape(x, shape)

	return x


def dot_product_attention(q, k, v, bias, training, dropout_rate=0.0, name=None):

	with tf.variable_scope(
			name,
			default_name="dot_product_attention",
			values=[q, k, v]):

		# [batch, num_heads, length_q, length_kv]
		logits = tf.matmul(q, k, transpose_b=True)

		if bias is not None:
			logits += bias

		weights = tf.nn.softmax(logits, name="attention_weights")

		# dropping out the attention links for each of the heads
		weights = tf.layers.dropout(weights, dropout_rate, training)

		return tf.matmul(weights, v)


def combine_heads_3d(x):

	return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 4, 1, 5]))


def combine_last_two_dimensions(x):

	old_shape = x.get_shape().dims
	a, b = old_shape[-2:]
	new_shape = old_shape[:-2] + [a * b if a and b else None]

	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
	ret.set_shape(new_shape)

	return ret
