from __future__ import absolute_import
import numpy as np
from network3 import *
from input_fn import *
from generate_tfrecordall import cut_edge, prepare_validation, load_subject
from utils import Golabal_Variable as GV

"""This script trains or evaluates the model.
"""
class Model(object):
    def __init__(self, sess, conf, results_acc_loss):
        self.conf = conf
        self.sess = sess
        self.result_acc_loss=results_acc_loss
        self.results_eval= {}
    def _model_fn(self, features, labels, mode):

        if self.conf.T == 'T1':
            self.conf.model_dir = './model-1'
            self.class_weights = tf.constant([[0.33, 1.5, 0.00, 0.00]])
        elif self.conf.T == 'T2':
            self.class_weights = tf.constant([[0.33, 0.00, 0.83, 0.00]])
            self.conf.model_dir = './model-2'
        else:
            self.class_weights = tf.constant([[0.33, 1.5, 0.83, 1.33]])
            self.conf.model_dir = './model-3'

        Model = Network(self.conf)
        logits = Model(features, mode == tf.estimator.ModeKeys.TRAIN)

        predictions = {
            'classes': tf.argmax(logits, axis=-1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))

        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        # Add weight decay to the loss.
        loss = cross_entropy + self.conf.weight_decay * tf.add_n(
             [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if 'kernel' in v.name ])

        # Weighted loss entropy loss
        loss = tf.reduce_mean(loss + tf.reduce_mean(self.class_weights * logits, axis=-1))
        #unweighted_losses = weights + self.conf.weight_decay



        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.compat.v1.train.get_or_create_global_step()
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.conf.learning_rate)

            # Batch norm requires update ops to be added as a dependency to train_op
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)

            self.sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        else:
            train_op = None

        accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])
        metrics = {'accuracy': accuracy}

        # Create a tensor named train_accuracy for logging purposes
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])

        # Create a tensor named loss for draw purposes.
        tf.identity(loss, name='loss')
        tf.summary.scalar('loss', loss)

        # create histogram of class spread
        tf.summary.histogram("classes", labels)

        self.layer_name=[v for v in tf.compat.v1.trainable_variables()]
        self.saver = tf.compat.v1.train.Saver()
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=metrics)

    def train(self):
        # Using the Winograd non-fused algorithms provides a small performance boost.
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        save_checkpoints_steps = self.conf.epochs_per_eval * \
                                 self.conf.num_training_subs // self.conf.batch_size
        run_config = tf.estimator.RunConfig().replace(
                                save_checkpoints_steps=save_checkpoints_steps,
                                keep_checkpoint_max=0)

        classifier = tf.estimator.Estimator(
                                    model_fn=self._model_fn,
                                model_dir=self.conf.model_dir,
                                config=run_config)
        tf.logging.info("train_epochs = {}, num_itreation = {}, batch_size = {}"
                        .format(self.conf.train_epochs,
                                self.conf.train_epochs // self.conf.epochs_per_eval,
                                self.conf.batch_size))
        z=0
        for tr in range(self.conf.train_epochs // self.conf.epochs_per_eval):
            tensors_to_log = {
                            'cross_entropy': 'cross_entropy',
                            'train_accuracy': 'train_accuracy'}
            logging_hook = tf.compat.v1.train.LoggingTensorHook(
                            tensors=tensors_to_log, every_n_iter=100)
            print('\n Starting a training cycle.')
            def input_fn_train():
                            return input_function(T=self.conf.T,
                                      data_dir=self.conf.data_dir,
                                      mode='train',
                                      patch_size=self.conf.patch_size,
                                      batch_size=self.conf.batch_size,
                                      buffer_size=self.conf.num_training_subs,
                                      valid_id=self.conf.validation_id,
                                      pred_id=-1,  # not used
                                      overlap_step=-1,  # not used
                                      num_epochs=self.conf.epochs_per_eval,
                                      num_parallel_calls=self.conf.num_parallel_calls)
            classifier.train(input_fn=input_fn_train, hooks=[logging_hook])
            if self.conf.T == 'T3':
                with self.sess:
                    GV.Wlist4 = {}
                    i = 0
                    for j in self.layer_name:
                        GV.Wlist4[j.name] =self.set_alfa()*classifier.get_variable_value(j.name) + \
                                           (1-self.set_alfa())*(GV.Wlist1[j.name] + GV.Wlist2[j.name])
                        var = [v for v in self.layer_name if v.name == j.name]
                        var = var[0].assign(self.layer_name[i])
                        self.sess.run(var,feed_dict = {var: GV.Wlist4[j.name]})
                        i +=1

            if self.conf.validation_id != -1:
                print('\n Starting to evaluate.')
                def input_fn_eval():
                        return input_function(T=self.conf.T,
                            data_dir=self.conf.data_dir,
                            mode='valid',
                            patch_size=self.conf.patch_size,
                            batch_size=self.conf.batch_size,
                            buffer_size=-1,  # not used
                            valid_id=self.conf.validation_id,
                            pred_id=-1,  # not used
                            overlap_step=self.conf.overlap_step,
                            num_epochs=1,
                            num_parallel_calls=self.conf.num_parallel_calls)

                self.results_eval=classifier.evaluate(input_fn=input_fn_eval)
                z=z+1
                print('results_eval=',self.results_eval['accuracy'],self.results_eval['loss'],'itration=',tr)
                #self.result_acc_loss.loc[z, :] = (np.round(self.results_eval['accuracy'], 2), np.round(self.results_eval['loss'], 2), tr)
                self.result_acc_loss.loc[z, :] = (self.results_eval['accuracy'], self.results_eval['loss'], tr)
        file_eval = 'evaluation_'+str(1)+'.csv'
        self.result_acc_loss.to_csv(self.conf.model_dir+'/' + file_eval)
        print('Number of parameters: ', len(classifier.get_variable_names()))
        Wdic = self.layer_name
        if self.conf.T == 'T1':
            for j in Wdic:
                GV.Wlist1[j.name] = classifier.get_variable_value(j.name)#.flatten()
        elif self.conf.T == 'T2':
            for j in Wdic:
                GV.Wlist2[j.name] = classifier.get_variable_value(j.name)#.flatten()
                GV.Wlist3[j.name] = (GV.Wlist1[j.name]+ GV.Wlist2[j.name]) #np.dstack((GV.Wlist1[j], GV.Wlist2[j]))
        else:
            print('the third model trained on all weigths')

    def predict(self):
        # Using the Winograd non-fused algorithms provides a small performance boost.
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        print('Perform prediction for subject-%d:' % self.conf.prediction_id)

        print('Loading data...')
        print('T=', self.conf.T)
        if self.conf.T == 'T1':
            [T1, _] = load_subject(self.conf.raw_data_dir, self.conf.prediction_id, self.conf.T)
        elif self.conf.T == 'T2':
            [T1,_] = load_subject(self.conf.raw_data_dir, self.conf.prediction_id, self.conf.T)
        else:
            [T1,_,_] = load_subject(self.conf.raw_data_dir, self.conf.prediction_id, self.conf.T)

        (org_zise, cut_size) = cut_edge(T1)
        print('original_size=',org_zise)
        print('Check cut_size: ', cut_size)

        cutted_T1 = T1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :]
        patch_ids = prepare_validation(cutted_T1, self.conf.patch_size, self.conf.overlap_step)
        num_patches = len(patch_ids)
        print('Number of patches:', num_patches,patch_ids)

        print('Initialize...')
        classifier = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.conf.model_dir)

        def input_fn_predict():
                return input_function(T=self.conf.T,
                    data_dir=self.conf.data_dir,
                    mode='pred',
                    patch_size=self.conf.patch_size,
                    batch_size=self.conf.batch_size,
                    buffer_size=-1,  # not used
                    valid_id=-1,  # not used
                    pred_id=self.conf.prediction_id,
                    overlap_step=self.conf.overlap_step,
                    num_epochs=1,
                    num_parallel_calls=self.conf.num_parallel_calls)

        checkpoint_file=tf.train.latest_checkpoint(
            self.conf.model_dir, latest_filename=None)

        preds = classifier.predict(input_fn= input_fn_predict,
                                    checkpoint_path=checkpoint_file)
        print('--------------------------------')
        print('Starting to predict.')

        predictions = {}
        for i, pred in enumerate(preds):
            location = patch_ids[i]
            print('Step {:d}/{:d} processing results for ({:d},{:d},{:d})'.format(
                i + 1, num_patches, location[0], location[1], location[2]),
                                end='\r',
                                flush=True)
            logits = pred['probabilities']
            #print('logits=',logits)
            for j in range(self.conf.patch_size):
                for k in range(self.conf.patch_size):
                    for l in range(self.conf.patch_size):
                        key = (location[0] + j, location[1] + k, location[2] + l)
                        if key not in predictions.keys():
                            predictions[key] = []
                        predictions[key].append(logits[j, k, l, :])
        print('Averaging results...')

        results = np.zeros((T1.shape[0], T1.shape[1], T1.shape[2], self.conf.num_classes),
                           dtype=np.float32)
        print(results.shape)
        for key in predictions.keys():
            results[cut_size[0] + key[0], cut_size[2] + key[1], cut_size[4] + key[2]] = \
                np.mean(predictions[key], axis=0)
        results = np.argmax(results, axis=-1)

        print('Saving results...')

        if not os.path.exists(self.conf.save_dir):
            os.makedirs(self.conf.save_dir)
        save_filename = 'preds-' + str(self.conf.checkpoint_num) + \
                        '-sub-' + str(self.conf.prediction_id) + \
                        '-overlap-' + str(self.conf.overlap_step) + \
                        '-patch-' + str(self.conf.patch_size) + '.npy'
        save_file = os.path.join(self.conf.save_dir, save_filename)
        np.save(save_file, results)

        print('Done.')

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.sess.graph)
        writer.close()

    def set_alfa(self):
        if len(self.results_eval)==0:
            self.conf.alfa=0.01
        else:
            self.conf.alfa=(self.results_eval['accuracy']+1)\
                           /((1/self.results_eval['loss'])+self.conf.batch_size)
        return self.conf.alfa