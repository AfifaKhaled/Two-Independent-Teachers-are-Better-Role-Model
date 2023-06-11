from configure import conf
import os
import tensorflow as tf
import pandas as pd
from model import *
import argparse
import logging
import numpy as np
from utils import Golabal_Variable as GV
import time
def main(_):
    results_acc_loss = pd.DataFrame(columns=['accrucey', 'Loss', 'itretion'], index=range(0,conf.train_epochs))
    print(results_acc_loss.shape)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=conf.gpu_frac)
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='predict',
                        help='actions: train or predict')
    args = parser.parse_args()
    if args.option not in ['train', 'predict']:
        print('invalid option: ', args.option)
        print("Please input a option: train or predict")
    elif args.option =='predict':
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            with tf.compat.v1.variable_scope('Model1'):
                conf.T = 'T1'
                conf.save_dir = './results1'
                conf.model_dir = './model-1'
                model1 = Model(sess, conf, results_acc_loss)
                model1.graph = sess.graph
                if args.option=='prediction':
                    print('\n',"Training the First Model",'\n')
                    getattr(model1, args.option)()
                else:
                    print('\n', "prediction the First Model", '\n')
                    print(time.time())
                    getattr(model1, args.option)()
                    print(time.time())
        with tf.compat.v1.Session(config=config) as sess:
            with tf.compat.v1.variable_scope('Model2'):
                conf.T = 'T2'
                conf.save_dir = './results2'
                conf.model_dir = './model-2'
                model2 = Model(sess, conf, results_acc_loss)
                model2.graph = sess.graph
                if args.option =='prediction':
                    print('\n',"Training for the Second Model",'\n')
                    getattr(model2, args.option)()
                else:
                    print('\n', "prediction for the Second Model", '\n')
                    getattr(model2, args.option)()
        with tf.compat.v1.Session(config=config) as sess:
            with tf.compat.v1.variable_scope('Model3'):
                conf.save_dir = './results3'
                conf.T = 'T3'
                conf.model_dir = './model-3'
                model3 = Model(sess, conf, results_acc_loss)
                model3.graph = sess.graph
                if args.option=='train':
                    print('\n',"Training the third Model",'\n')
                    getattr(model3, args.option)()
                else:
                    print('\n', "prediction the third Model", '\n')
                    getattr(model3, args.option)()
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    LOG = logging.getLogger('main')
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.compat.v1.app.run()
