#import tensorflow as tf

from utils import *
"""This script defines the network.
"""
class Network(object):

	def __init__(self, conf,reuse=tf.AUTO_REUSE):
		# configure
		self.num_classes = conf.num_classes
		self.num_filters = conf.num_filters
		self.block_sizes = [1] * conf.network_depth
		self.block_strides = [1] + [2] * (conf.network_depth - 1)
		self.T=conf.T

	def __call__(self, inputs, training):
		"""Add operations to classify a batch of input images.

		Args:
			inputs: A Tensor representing a batch of input images.
			training: A boolean. Set to True to add operations required only when
				training the classifier.

		Returns:
			A logits Tensor with shape [<batch_size>, self.num_classes].
		"""

		return self._build_network(inputs, training)
	################################################################################
	# Composite blocks building the network
	################################################################################
	def _build_network(self, inputs, training):
		"""Build the network.
		"""
		inputs = Conv3D(inputs=inputs,
					filters=self.num_filters,
					kernel_size=3,
					strides=1)
		inputs = tf.identity(inputs, 'initial_conv')

		skip_inputs = []
		j=0
		for i, num_blocks in enumerate(self.block_sizes):
			num_filters = self.num_filters * (2**i)
			inputs = self._encoding_block_layer(j,
						inputs=inputs, filters=num_filters,
						block_fn=self._residual_block, blocks=num_blocks,
						strides=self.block_strides[i], training=training,
						name='encode_block_layer{}'.format(i+1))
			skip_inputs.append(inputs)
			j+=1

			#print('inputs pool=',inputs)
		inputs = BN_ReLU(inputs, training)
		num_filters = self.num_filters * (2**(len(self.block_sizes)-1))
		output_shape = skip_inputs[-1].get_shape()
		#inputs = Pool3d(inputs, 1, 1)
		inputs = multihead_attention_3d(
					inputs, num_filters, num_filters, num_filters, 2, training,output_shape, layer_type='SAME')
		#inputs = Pool3d(inputs, 1, 1)
		inputs += skip_inputs[-1]
		for i, num_blocks in reversed(list(enumerate(self.block_sizes[1:]))):
			num_filters = self.num_filters * (2 **i)
			if i == len(self.block_sizes) - 1:
				inputs = self._att_decoding_block_layer(j,
						inputs=inputs, skip_inputs=skip_inputs[i],
						filters=num_filters, block_fn=self._residual_block,
						blocks=1, strides=self.block_strides[i+1],
						training=training,
						name='decode_block_layer{}'.format(len(self.block_sizes)-i-1))
				j += 1
			else:
				inputs = self._decoding_block_layer(j,
						inputs=inputs, skip_inputs=skip_inputs[i],
						filters=num_filters, block_fn=self._residual_block,
						blocks=1, strides=self.block_strides[i+1],
						training=training,
						name='decode_block_layer{}'.format(len(self.block_sizes)-i-1),reuse=True)
				j += 1
		inputs = self._output_block_layer(inputs=inputs, training=training)
		return inputs

	################################################################################
	# Composite blocks building the network
	################################################################################
	def _output_block_layer(self, inputs, training):
		inputs = BN_ReLU(inputs, training)

		inputs = tf.layers.dropout(inputs, rate=0.5, training=training)
		inputs = Conv3D(inputs=inputs,
					filters=self.num_classes,
					kernel_size=1,
					strides=1,name='convd_out',
					use_bias=True)

		return tf.identity(inputs, 'output')


	def _encoding_block_layer(self, j,inputs, filters, block_fn,
								blocks, strides, training, name):
		"""Creates one layer of encoding blocks for the model.

		Args:
			inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
			filters: The number of filters for the first convolution of the layer.
			block_fn: The block to use within the model.
			blocks: The number of blocks contained in the layer.
			strides: The stride to use for the first convolution of the layer. If
				greater than 1, this layer will ultimately downsample the input.
			training: Either True or False, whether we are currently training the
				model. Needed for batch norm.
			name: A string name for the tensor output of the block layer.

		Returns:
			The output tensor of the block layer.
		"""

		def projection_shortcut(inputs):
			return Conv3D(inputs=inputs,
								filters=filters,
								kernel_size=1,
								strides=strides,
							  name="convd3"+str(j))

		# Only the first block per block_layer uses projection_shortcut and strides
		inputs = block_fn(j,inputs, filters, training, projection_shortcut, strides)
		for _ in range(1, blocks):
			inputs = block_fn(j,inputs, filters, training, None, 1)
		return tf.identity(inputs, name)


	def _att_decoding_block_layer(self,j, inputs, skip_inputs, filters,
								block_fn, blocks, strides, training, name):
		"""Creates one layer of decoding blocks for the model.

		Args:
			inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
			skip_inputs: A tensor of size [batch, depth_in, height_in, width_in, filters].
			filters: The number of filters for the first convolution of the layer.
			block_fn: The block to use within the model.
			blocks: The number of blocks contained in the layer.
			strides: The stride to use for the first convolution of the layer. If
				greater than 1, this layer will ultimately downsample the input.
			training: Either True or False, whether we are currently training the
				model. Needed for batch norm.
			name: A string name for the tensor output of the block layer.

		Returns:
			The output tensor of the block layer.
		"""
		output_shape=skip_inputs.get_shape().as_list()
		def projection_shortcut(inputs):
			return Deconv3D(inputs=inputs,
								out_shape=output_shape,
								filters=filters,
								kernel_size=3,
								strides=strides,name='dconv3d1')

		inputs = self._attention_block(
					inputs, filters, training, projection_shortcut, strides,output_shape)
		inputs = inputs + skip_inputs

		for _ in range(0, blocks):
			inputs = block_fn(j,inputs, filters, training, None, 1,reuse=True)

		return tf.identity(inputs, name)


	def _decoding_block_layer(self,j, inputs, skip_inputs, filters,
								block_fn, blocks, strides, training, name,reuse=False):
		"""Creates one layer of decoding blocks for the model.

		Args:
			inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
			skip_inputs: A tensor of size [batch, depth_in, height_in, width_in, filters].
			filters: The number of filters for the first convolution of the layer.
			block_fn: The block to use within the model.
			blocks: The number of blocks contained in the layer.
			strides: The stride to use for the first convolution of the layer. If
				greater than 1, this layer will ultimately downsample the input.
			training: Either True or False, whether we are currently training the
				model. Needed for batch norm.
			name: A string name for the tensor output of the block layer.

		Returns:
			The output tensor of the block layer.
		"""
		output_shape = skip_inputs.get_shape().as_list()
		inputs = Deconv3D(inputs=inputs,
							out_shape=output_shape,
							filters=filters,
							kernel_size=3,
							strides=strides,name='dconv3d2'+str(j))

		inputs = inputs + skip_inputs
		for _ in range(0, blocks):
			inputs = block_fn(j,inputs, filters, training, None, 1)
		return tf.identity(inputs, name)


	################################################################################
	# Basic blocks building the network
	################################################################################
	def _residual_block(self,j, inputs, filters, training,
							projection_shortcut, strides,reuse=False):
		"""Standard building block for residual networks with BN before convolutions.

		Args:
			inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
			filters: The number of filters for the convolutions.
			training: A Boolean for whether the model is in training or inference
				mode. Needed for batch normalization.
			projection_shortcut: The function to use for projection shortcuts
				(typically a 1x1 convolution when downsampling the input).
			strides: The block's stride. If greater than 1, this block will ultimately
				downsample the input.

		Returns:
			The output tensor of the block.
		"""
		shortcut = inputs
		inputs = BN_ReLU(inputs, training)

		if projection_shortcut is not None:
			shortcut = projection_shortcut(inputs)
		inputs = Conv3D(inputs=inputs,
					filters=filters,
					kernel_size=3,
					strides=strides,name="convd32"+str(j))

		inputs = BN_ReLU(inputs, training)
		inputs = Conv3D(inputs=inputs,
					filters=filters,
					kernel_size=3,
					strides=1,name="convd33"+str(j),reuse=reuse)

		return inputs + shortcut

	def _attention_block(self, inputs, filters, training,
							projection_shortcut, strides,output_shape):
		"""Attentional building block for residual networks with BN before convolutions.

		Args:
			inputs: A tensor of size [batch, depth_in, height_in, width_in, channels].
			filters: The number of filters for the convolutions.
			training: A Boolean for whether the model is in training or inference
				mode. Needed for batch normalization.
			projection_shortcut: The function to use for projection shortcuts
				(typically a 1x1 convolution when downsampling the input).
			strides: The block's stride. If greater than 1, this block will ultimately
				downsample the input.

		Returns:
			The output tensor of the block.
		"""

		shortcut = inputs
		inputs = BN_ReLU(inputs, training)

		if projection_shortcut is not None:
			shortcut = projection_shortcut(inputs)

		if strides != 1:
			layer_type = 'UP'
		else:
			layer_type = 'SAME'

		#shortcut = Pool3d(shortcut, 1, 1)
		#inputs = Pool3d(inputs, 1, 1)
		inputs = multihead_attention_3d(
					inputs, filters, filters, filters, 1, training,output_shape, layer_type)

		return inputs + shortcut
