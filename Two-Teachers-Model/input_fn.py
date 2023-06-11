import tensorflow as tf
import os
from configure import *

"""This script defines the input interface.
"""
################################################################################
# Functions
################################################################################
def get_filenames(T,data_dir, mode, valid_id, pred_id, overlap_step, patch_size):
	"""Returns a list of filenames."""

	if mode == 'train':
		train_files = [
			os.path.join(data_dir, 'subject-%d.tfrecords' % i)

			for i in range(1, 11)
			if i != valid_id
		]
		#print(train_files)
		for f in train_files:
			assert os.path.isfile(f), \
				('Run generate_tfrecord.py to generate training files.')
		return train_files
	elif mode == 'valid':
		valid_file = os.path.join(data_dir, 
			'subject-%d-valid-%d-patch-%d.tfrecords' % (valid_id, overlap_step, patch_size))
		assert os.path.isfile(valid_file), \
			('Run generate_tfrecord.py to generate the validation file.')
		return [valid_file]
	elif mode == 'pred':
		print('T for data=',T)
		pred_file = os.path.join(data_dir,
			'subject-%d-pred-%d-patch-%d.tfrecords' % (pred_id, overlap_step, patch_size))
		assert os.path.isfile(pred_file), \
			('Run generate_tfrecord.py to generate the prediction file.')
		return [pred_file]


def decode_train(serialized_example):
	"""Parses training data from the given `serialized_example`."""
	if conf.T =='T1':

		features = tf.parse_single_example(
			serialized_example,
			features = {
			'T1': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.string),
			'original_shape': tf.FixedLenFeature(3, tf.int64),
			'cut_size': tf.FixedLenFeature(6, tf.int64)
		   })
	elif  conf.T =='T2':
		features = tf.parse_single_example(
			serialized_example,
			features={
				'T2':tf.FixedLenFeature([], tf.string),
				'label':tf.FixedLenFeature([],tf.string),
				'original_shape':tf.FixedLenFeature(3, tf.int64),
				'cut_size':tf.FixedLenFeature(6, tf.int64)
		    })
	else:
		features = tf.parse_single_example(
			serialized_example,
			features = {
			'T1': tf.FixedLenFeature([], tf.string),
			'T2':tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.string),
			'original_shape': tf.FixedLenFeature(3, tf.int64),
			'cut_size': tf.FixedLenFeature(6, tf.int64)
		   })
	#features = tf.parse_single_example(serialized_example,features=features)

	img_shape = features['original_shape']
	cut_size = features['cut_size']

	# Convert from a scalar string tensor
	if  conf.T =='T1':
		image_T1 = tf.decode_raw(features['T1'], tf.int16)
		image_T1 = tf.reshape(image_T1, img_shape)
		image_T1 = tf.cast(image_T1, tf.float32)

	elif  conf.T =='T2':
		image_T2 = tf.decode_raw(features['T2'], tf.int16)
		image_T2 = tf.reshape(image_T2, img_shape)
		image_T2 = tf.cast(image_T2, tf.float32)
	else:
		image_T1 = tf.decode_raw(features['T1'], tf.int16)
		image_T1 = tf.reshape(image_T1, img_shape)
		image_T2 = tf.decode_raw(features['T2'], tf.int16)
		image_T2 = tf.reshape(image_T2, img_shape)
		# Convert dtype.
		image_T1 = tf.cast(image_T1, tf.float32)
		image_T2 = tf.cast(image_T2, tf.float32)
	label = tf.decode_raw(features['label'], tf.uint8)
	label = tf.reshape(label, img_shape)
	label = tf.cast(label, tf.float32)
	if conf.T=='T1':
			return image_T1,  label, cut_size
	elif conf.T=='T2':
		return image_T2, label, cut_size
	else:
		return image_T1,image_T2, label, cut_size  #


def decode_valid(serialized_example):
	"""Parses validation data from the given `serialized_example`."""
	if conf.T=='T1':
		features = {
				'T1':tf.FixedLenFeature([],tf.string),
				'label':tf.FixedLenFeature([],tf.string)
					}
	elif conf.T=='T2':
		features={
				'T2':tf.FixedLenFeature([], tf.string),
				'label':tf.FixedLenFeature([],tf.string)
					}
	else:
		features = {
				'T1':tf.FixedLenFeature([],tf.string),
				'T2':tf.FixedLenFeature([], tf.string),
				'label':tf.FixedLenFeature([],tf.string)
					}
	features = tf.parse_single_example(serialized_example,features=features)

	patch_shape = [conf.patch_size, conf.patch_size, conf.patch_size]

	# Convert from a scalar string tensor
	if  conf.T =='T1':
		image_T1 = tf.decode_raw(features['T1'], tf.int16)
		image_T1 = tf.reshape(image_T1, patch_shape)
		image_T1 = tf.cast(image_T1, tf.float32)

	elif  conf.T =='T2':
		image_T2 = tf.decode_raw(features['T2'], tf.int16)
		image_T2 = tf.reshape(image_T2, patch_shape)
		image_T2 = tf.cast(image_T2, tf.float32)

	else:
		image_T1 = tf.decode_raw(features['T1'], tf.int16)
		image_T1 = tf.reshape(image_T1, patch_shape)
		image_T2 = tf.decode_raw(features['T2'], tf.int16)
		image_T2 = tf.reshape(image_T2, patch_shape)
		# Convert dtype.
		image_T1 = tf.cast(image_T1, tf.float32)
		image_T2 = tf.cast(image_T2, tf.float32)
	label = tf.decode_raw(features['label'], tf.uint8)
	label = tf.reshape(label, patch_shape)

	if conf.T=='T1':
			return image_T1,  label
	elif conf.T=='T2':
		return image_T2, label
	else:
		return image_T1,image_T2, label

def decode_pred(serialized_example):
	"""Parses prediction data from the given `serialized_example`."""
	if  conf.T  == 'T1':
		features = {
			'T1': tf.FixedLenFeature([], tf.string),
		}
	elif conf.T == 'T2':
		features={
				'T2':tf.FixedLenFeature([], tf.string),
					}
	else:
		features = {
				'T1':tf.FixedLenFeature([],tf.string),
				'T2':tf.FixedLenFeature([], tf.string),
					}
	features = tf.parse_single_example(
					serialized_example,
					features=features)

	patch_shape = [conf.patch_size, conf.patch_size, conf.patch_size]

	# Convert from a scalar string tensor
	if  conf.T =='T1':
		image_T1 = tf.decode_raw(features['T1'], tf.int16)
		image_T1 = tf.reshape(image_T1, patch_shape)
		# Convert dtype.
		image_T1 = tf.cast(image_T1, tf.float32)
		label = tf.zeros(image_T1.shape)  # pseudo label
	elif  conf.T =='T2':
		image_T2 = tf.decode_raw(features['T2'], tf.int16)
		image_T2 = tf.reshape(image_T2, patch_shape)
		# Convert dtype.
		image_T2 = tf.cast(image_T2, tf.float32)
		label = tf.zeros(image_T2.shape)  # pseudo label
	else:
		image_T1 = tf.decode_raw(features['T1'], tf.int16)
		image_T1 = tf.reshape(image_T1, patch_shape)
		image_T2 = tf.decode_raw(features['T2'], tf.int16)
		image_T2 = tf.reshape(image_T2, patch_shape)
		# Convert dtype.
		image_T1 = tf.cast(image_T1, tf.float32)
		image_T2 = tf.cast(image_T2, tf.float32)
		label = tf.zeros(image_T1.shape)  # pseudo label

	if conf.T =='T1':
			return image_T1,  label
	elif conf.T=='T2':
		return image_T2, label
	else:
		return image_T1,image_T2, label


def crop_image(image_T, label, cut_size): #
	"""Crop training data."""
	# Randomly crop a [patch_size, patch_size, patch_size] section of the image.

	data = tf.stack([image_T, label], axis=-1) #
	image = tf.random_crop(
		data[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :],
		[conf.patch_size, conf.patch_size, conf.patch_size, 2])

	[image_T, label] = tf.unstack(image, 2, axis=-1)  # image_T2,

	return image_T, label


def crop_image2(image_T1,image_T2, label, cut_size):  #

	data = tf.stack([image_T1,image_T2, label], axis=-1) #
	image = tf.random_crop(
		data[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :],
		[conf.patch_size, conf.patch_size, conf.patch_size, 3])

	[image_T1,image_T2, label] = tf.unstack(image, 3, axis=-1)  #

	return image_T1, image_T2, label


def normalize_image(image_T, label): #
	"""Normalize data."""
	# Subtract off the mean and divide by the variance of the pixels.
	image_T = tf.image.per_image_standardization(image_T)
	features = tf.stack([image_T], axis=-1) #, image_T2
	label = tf.cast(label, tf.int32)
	return features, label


def normalize_image1(image_T1, image_T2, label):  #
	"""Normalize data."""
	# Subtract off the mean and divide by the variance of the pixels.
	image_T1 = tf.image.per_image_standardization(image_T1)
	image_T2 = tf.image.per_image_standardization(image_T2)
	features = tf.stack([image_T1, image_T2], axis=-1)  #
	label = tf.cast(label, tf.int32)

	return features, label


def input_function(T,data_dir, mode, patch_size, batch_size, buffer_size, valid_id,
				   pred_id, overlap_step, num_epochs=1, num_parallel_calls=1):

	with tf.name_scope('input'):
		# Generate a Dataset with raw records.
		filenames = get_filenames(T,data_dir, mode, valid_id, pred_id, overlap_step, patch_size)
		dataset = tf.data.TFRecordDataset(filenames)

		# We prefetch a batch at a time, This can help smooth out the time taken to
		# load input files as we go through shuffling and processing.
		dataset = dataset.prefetch(buffer_size=batch_size)
		if mode == 'train':
			# Shuffle the records. Note that we shuffle before repeating to ensure
			# that the shuffling respects epoch boundaries.
			dataset = dataset.shuffle(buffer_size=buffer_size)

		# If we are training over multiple epochs before evaluating, repeat the
		# dataset for the appropriate number of epochs.
		dataset = dataset.repeat(num_epochs)
		if (mode == 'train') and (conf.T=='T1' or conf.T=='T2'):
			dataset = dataset.map(decode_train, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(crop_image, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(normalize_image, num_parallel_calls=num_parallel_calls)

		elif (mode=='train') and (conf.T=='T3'):
			dataset = dataset.map(decode_train, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(crop_image2, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(normalize_image1, num_parallel_calls=num_parallel_calls)

		elif (mode == 'valid') and (conf.T=='T1' or conf.T=='T2'):
			dataset = dataset.map(decode_valid, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(normalize_image, num_parallel_calls=num_parallel_calls)
		elif (mode == 'valid') and (conf.T == 'T3'):
			dataset = dataset.map(decode_valid, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(normalize_image1, num_parallel_calls=num_parallel_calls)

		elif (mode == 'pred') and (conf.T=='T1' or conf.T=='T2'):
			dataset = dataset.map(decode_pred, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(normalize_image, num_parallel_calls=num_parallel_calls)
		elif(mode=='pred') and (conf.T=='T3'):
			print('T for prediction data=',T)
			dataset = dataset.map(decode_pred, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(normalize_image1, num_parallel_calls=num_parallel_calls)
		dataset = dataset.batch(batch_size, drop_remainder=True)
		dataset = dataset.prefetch(1)
		iterator = dataset.make_one_shot_iterator()
		features, label = iterator.get_next()
		return features, label
