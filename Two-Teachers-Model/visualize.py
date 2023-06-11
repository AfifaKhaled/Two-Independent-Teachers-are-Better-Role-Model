import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configure import conf
"""Visualize results by slices.
"""


################################################################################
# Arguments
################################################################################
conf=conf
RAW_DATA_DIR = conf.raw_data_dir
LABEL_DIR = conf.data_dir #'./Datasets'
PRED_DIR = conf.save_dir #'./results3'
PRED_ID = conf.prediction_id #11  # 1-10
PATCH_SIZE = conf.patch_size #32
CHECKPOINT_NUM =conf.checkpoint_num # 4
OVERLAP_STEPSIZE = conf.overlap_step #8
SLICE_DEPTH = 150

################################################################################
# Functions
################################################################################
def Visualize(label_dir, pred_dir, pred_id, patch_size, checkpoint_num,
		overlap_step, slice_depth):
	print('Perform visualization for subject-%d:' % pred_id)
	label_id=conf.validation_id
	print('Loading label...')
	label_file = os.path.join(label_dir, 'subject-%d-label.npy' % label_id)
	assert os.path.isfile(label_file), \
		('Run generate_tfrecord.py to generate the label file.')
	label = np.load(label_file)
	print('Check label: ', label.shape, np.max(label))

	print('Loading predition...')
	pred_file = os.path.join(pred_dir, 
				'preds-%d-sub-%d-overlap-%d-patch-%d.npy' % \
				(checkpoint_num, pred_id, overlap_step, patch_size))
	assert os.path.isfile(pred_file), \
		('Run main.py --option=predict to generate the prediction results.')
	pred = np.load(pred_file)
	print('Check pred: ', pred.shape, np.max(pred))

	pred_show = pred[:, :, slice_depth]
	label_show = label[:, :, slice_depth]

	fig = plt.figure()
	fig.suptitle('Compare the %d-th slice.' % slice_depth, fontsize=14)

	a = fig.add_subplot(1,2,1)
	imgplot = plt.imshow(label_show)
	a.set_title('Groud Truth')

	a = fig.add_subplot(1,2,2)
	imgplot = plt.imshow(pred_show)
	a.set_title('Prediction')

	plt.savefig('visualization-%d-sub-%d-overlap-%d+3' % \
			(checkpoint_num, pred_id, overlap_step))

if __name__ == '__main__':
	Visualize(
		label_dir=LABEL_DIR,
		pred_dir=PRED_DIR,
		pred_id=PRED_ID,
		patch_size=PATCH_SIZE,
		checkpoint_num=CHECKPOINT_NUM,
		overlap_step=OVERLAP_STEPSIZE,
		slice_depth=SLICE_DEPTH)