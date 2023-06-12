import tensorflow as tf
"""This script defines hyperparameters.
"""
def configure():
	flags =tf.compat.v1.app.flags

	# Training set
	flags.DEFINE_string('raw_data_dir', './Datasets',
			'the directory where the raw data is stored')
	flags.DEFINE_string('data_dir', './Datasets',
			'the directory where the input data is stored')
	flags.DEFINE_integer('num_training_subs', 9,
			'the number of subjects used for training')
	flags.DEFINE_integer('train_epochs',300000,
			'the number of epochs to use for training')
	flags.DEFINE_integer('epochs_per_eval', 50,
			'the number of training epochs to run between evaluations')
	flags.DEFINE_integer('batch_size', 32,
			'the number of examples processed in each training batch')
	flags.DEFINE_float('learning_rate', 3e-5, 'learning rate')
	flags.DEFINE_float('weight_decay', 3e-5, 'weight decay rate')
	flags.DEFINE_integer('num_parallel_calls',2,
			'The number of records that are processed in parallel \
			during input processing. This can be optimized per data set but \
			for generally homogeneous data sets, should be approximately the \
			number of available CPU cores.')
	flags.DEFINE_float("alfa", 0.01, "alfa term of update weights (default: 0.01)")
	flags.DEFINE_string('model_dir', './model-1',
			'the directory where the model will be stored')
	flags.DEFINE_string('T', 'T1',
						 'Number of channels in input image(1 for T1 and 2 for T2 and 3 for all)')
	flags.DEFINE_float("gpu_frac", 0.95, "Gpu fraction if you have Gpu")

	# Validation set / Prediction set
	flags.DEFINE_integer('patch_size', 32, 'spatial size of patches')
	flags.DEFINE_integer('overlap_step', 8,
			'overlap step size when performing validation/prediction')
	flags.DEFINE_integer('validation_id', 8,
			'1-10 or -1, which subject is used for validation')
	flags.DEFINE_integer('prediction_id', 11,
			'1-23, which subject is used for prediction')
	flags.DEFINE_integer('checkpoint_num',4,
			'which checkpoint is used for validation/prediction')
	flags.DEFINE_string('save_dir', './results3',
			'the directory where the prediction is stored')

	# network
	flags.DEFINE_integer('network_depth', 3, 'the network depth')
	flags.DEFINE_integer('num_classes', 4, 'the number of classes')
	flags.DEFINE_integer('num_filters', 32, 'number of filters for initial_conv')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS


conf = configure()