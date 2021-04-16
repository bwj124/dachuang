#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import sys
import json
import numpy as np
import pickle
import random
import re
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras.backend import set_session

from utils import MLP, ThreadedIterator, SMALL_NUMBER


class GGNN(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 100,#3000,
            'patience': 100,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 100,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 5827,

            'train_file': 'train_set_small.json',
            'valid_file': 'test_set_small.json',
            'timeout': 24*60*60,

            'predict': False
        }

    def __init__(self, args):
        init_time = time.time()
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
            self.data_dir = data_dir
        
        # The data for generating vector
        myalldata_dir = ''
        if '--myalldata_dir' in args and args['--myalldata_dir'] is not None:
            myalldata_dir = args['--myalldata_dir']
            self.myalldata_dir = myalldata_dir

        # Collect parameters:
        params = self.default_params()
        if '--train_file' in args and args['--train_file'] is not None:
            params['train_file'] = args['--train_file']
            self.train_file = params['train_file']

        if '--valid_file' in args and args['--valid_file'] is not None:
            params['valid_file'] = args['--valid_file']
            self.valid_file = params['valid_file']
        
        if '--predict' in args and args['--predict'] is not None:
            params['predict'] = args['--predict']

        if '--epoch' in args and args['--epoch'] is not None:
            params['num_epochs'] = int(args['--epoch'])

        
        log_sub_path = re.sub(r'_\d+', '', params['train_file'].replace('train_', '').replace('.json', ''))
        # if '--random_seed' in args and args['--random_seed'] is not None:
        #     params['random_seed'] = int(args['--random_seed'])
        
        # self.run_id = "_".join([params['train_file'], time.strftime("%Y-%m-%d-%H-%M-%S")])
        self.run_id = params['train_file']
        log_dir = args.get('--log_dir') or 'logs/{}'.format(log_sub_path)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)
        self.online_data_backup_file = os.path.join(log_dir, "%s_result" % self.run_id)
        
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])
        args_time = time.time()

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'], is_training_data=False)
        loaddata_time = time.time()
        
        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config_time = time.time()
        
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        #self.sess = tf.InteractiveSession(graph=self.graph, config=config)
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()
            make_model_time = time.time()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()
        graph_sess_time = time.time()
        
        out_str = "load args: {}\t load data time: {}\t config time: {}\t make_model time: {}\t build(except make_model) time: {}".format(args_time-init_time, loaddata_time-args_time, config_time-loaddata_time, make_model_time-config_time, graph_sess_time-make_model_time)
        print(out_str)
        with open('./outputs/time.log', 'a') as f:
            f. write('---------- make_model: '+str(int(init_time))+'----------\n')
            f.write(out_str+'\n')
            

    def is_predict(self):
        return self.params['predict']

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
        except:
            os.exit()
            print('load...')

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            if len(g['graph']) == 0:
                continue
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [int(e[0]), int(e[2])]]))
        #     num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        # max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        # self.max_num_vertices = 16479
        self.num_edge_types = 33
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        num_task_id = len(self.params['task_ids'])
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [num_task_id, None, 2*num_task_id], name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [num_task_id, None, 2*num_task_id], name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            set_session(self.sess)
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
                #print(self.ops['final_node_representations'].shape)
                #zero_array = np.zeros(self.ops['final_node_representations'].shape)
                #sess=tf.Session()
                #x = self.ops['final_node_representations'].eval(session=sess,feed_dict={self.ops['final_node_representations']:zero_array})
                #print(x)
                #with tf.Session() as sess:
                #    vector = self.ops['final_node_representations'].eval(session=sess)
                #    print(vector)
                # with open ('./outputs/ggnn_vector.txt', 'a') as f:
                #     f.write(str(tf.Session().run(self.ops['final_node_representations'])))
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 2, [],
                                                                           self.placeholders['out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 2, [],
                                                                                self.placeholders['out_layer_dropout_keep_prob'])
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id])

                #computed_values = tf.Print(computed_values-0.5, [computed_values-0.5, tf.shape(computed_values)], 'computed_values', summarize = 150)
                tv = self.placeholders['target_values'][internal_id,:] #tf.squeeze(
                #tv = tf.Print(tv, [tv, tf.shape(tv)], 'tv', summarize = 150)
                # if computed_values.shape.as_list() == tv.shape.as_list():
                #     tv = tf.squeeze(tv)
                #with open('labels_computedValues.txt','a') as f:
                #    f.write('target_values:'+str(self.sess.run(self.tv))+'\ncomputed_values:'+str(self.sess.run(self.computed_values))+'\n')
                labels = tf.argmax(tv, 1)
                prediction = tf.argmax(computed_values, 1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))
                task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=computed_values, labels=tv))

                self.tv = tv
                self.computed_values = computed_values
                self.labels = labels
                self.prediction = prediction

                TP = tf.reduce_sum(prediction*labels)
                TN = tf.reduce_sum((1-prediction)*(1-labels))
                FP = tf.reduce_sum(prediction*(1-labels))
                FN = tf.reduce_sum((1-prediction)*labels)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)

                self.ops['TP%i' % task_id] = TP
                self.ops['TN%i' % task_id] = TN
                self.ops['FP%i' % task_id] = FP
                self.ops['FN%i' % task_id] = FN

                self.ops['accuracy_task%i' % task_id] = accuracy
                self.ops['losses'].append(task_loss)

                self.ops['precision_task%i' % task_id] = precision
                self.ops['recall_task%i' % task_id] = recall
                self.ops['f1_task%i' % task_id] = f1
                
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def run_epoch(self, epoch_name: str, data, is_training: bool, is_test: bool):
        loss = 0
        TP_all = []
        TN_all = []
        FP_all = []
        FN_all = []
        #accuracies = []
        #precision = []
        #recall = []
        #f1=[]
        #accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        #precision_ops = [self.ops['precision_task%i' % task_id] for task_id in self.params['task_ids']]
        #recall_ops = [self.ops['recall_task%i' % task_id] for task_id in self.params['task_ids']]
        #f1_ops = [self.ops['f1_task%i' % task_id] for task_id in self.params['task_ids']]
        TP_ops = [self.ops['TP%i' % task_id] for task_id in self.params['task_ids']]
        TN_ops = [self.ops['TN%i' % task_id] for task_id in self.params['task_ids']]
        FP_ops = [self.ops['FP%i' % task_id] for task_id in self.params['task_ids']]
        FN_ops = [self.ops['FN%i' % task_id] for task_id in self.params['task_ids']]

        
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        
        for step, batch_data in enumerate(batch_iterator):
            
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                #fetch_list = [self.ops['loss'], accuracy_ops, accuracy_ops, precision_ops, recall_ops, f1_ops, self.ops['train_step']]
                fetch_list = [TP_ops, TN_ops, FP_ops, FN_ops, self.ops['loss']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                #fetch_list = [self.ops['loss'], accuracy_ops, precision_ops, recall_ops, f1_ops]
                fetch_list = [TP_ops, TN_ops, FP_ops, FN_ops, self.ops['loss']]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            #if is_training:
            #    train_vector = []
            #    target_vector = []
            #    res = self.sess.run([self.computed_values, self.prediction,self.tv,self.labels],feed_dict=batch_data)

                #res = self.sess.run([self.computed_values, self.tv],feed_dict=batch_data)
                #train_vector = res[0]
                #target_vector = res[1]

            #    computed_v = res[0]
            #    pred = res[1]
            #    target_v = res[2]
            #    labels = res[3]
            #    for i, array in enumerate(computed_v):
            #        train_vector.append(array[pred[i]])
            #    for i, array in enumerate(target_v):
            #        target_vector.append(array[labels[i]])
            #    with open('./outputs/train_vector.txt','a') as f:
            #        f.write(str(train_vector)+'\n')
            #    with open('./outputs/target_vector.txt','a') as f:
            #        f.write(str(target_vector)+'\n')
            #else:
                #vector = self.sess.run(self.computed_values,feed_dict=batch_data)
            #    vector = []
            #    for i, array in enumerate(self.sess.run(self.computed_values,feed_dict=batch_data)):
            #        indices = self.sess.run(self.prediction,feed_dict=batch_data)
            #        vector.append(array[indices[i]])
            #    with open('./outputs/vector.txt','a') as f:
            #        f.write(str(vector)+'\n')
            # gated_output = self.sess.run(self.gated_outputs,feed_dict=batch_data)
            if not is_training and is_test:
                final_node = self.sess.run(self.compute_final_node_representations(),feed_dict=batch_data)
                final_node = np.mean(final_node, 0)
                np.save('./outputs/ggnn_vector/{}_test_final_node.npy'.format(self.valid_file), final_node)
            # np.save('./outputs/gated_output.npy', gated_output)

            #with open('./outputs/gated_outputs.txt', 'a') as f:
            #    f.write(str(gated_output)+'\n')

            (TP_batch, TN_batch, FP_batch, FN_batch, batch_loss) = (result[0], result[1], result[2], result[3], result[4]) 
            TP_all.append(TP_batch)
            TN_all.append(TN_batch)
            FP_all.append(FP_batch)
            FN_all.append(FN_batch)
            #(batch_loss, batch_accuracies, batch_precision, batch_recall, batch_f1) = (result[0], result[1], result[2], result[3], result[4])
            loss += batch_loss * num_graphs
            #accuracies.append(np.array(batch_accuracies) * num_graphs)
            #precision.append(np.array(batch_precision) * num_graphs)
            #recall.append(np.array(batch_recall) * num_graphs)
            #f1.append(np.array(batch_f1) * num_graphs)

        TP_all = tf.reduce_sum(TP_all)
        TN_all = tf.reduce_sum(TN_all)
        FP_all = tf.reduce_sum(FP_all)
        FN_all = tf.reduce_sum(FN_all)
        #accuracies = np.sum(accuracies, axis=0) / processed_graphs
        #precision = np.sum(precision, axis=0) / processed_graphs
        #recall = np.sum(recall, axis=0) / processed_graphs
        #f1 = np.sum(f1, axis=0) / processed_graphs
        loss = loss / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        accuracies = (TP_all + TN_all)/(TP_all + TN_all + FP_all + FN_all)
        precision = (TP_all)/(TP_all + FP_all)
        recall = (TP_all)/(TP_all + FN_all)
        f1 = 2 * precision * recall / (precision + recall)
        print(epoch_name)
        #with tf.Session() as sess:
        #    accuracies = accuracies.eval()
        #    precision = precision.eval()
        #    recall = recall.eval()
        #    f1 = f1.eval()

        accuracies = float(self.sess.run(accuracies))
        precision = float(self.sess.run(precision))
        recall = float(self.sess.run(recall))
        f1 = float(self.sess.run(f1))

        #accuracies = list(accuracies.numpy())
        #precision = list(precision.numpy())
        #recall = list(recall.numpy())
        #f1 = list(f1.numpy())

        return loss, accuracies, precision, recall, f1, instance_per_sec

    def train(self, is_test):
        train_begin_time = time.time()
        log_to_save = []
        total_time_start = time.time()
        summ_line = '%d\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f'
        line = 'loss:%.2f\tacc:%.2f\tprecision:%.2f\trecall:%.2f\tf1:%.2f\tspeed:%.2f'

        bak_train_data = []
        bak_valid_data = []
        with self.graph.as_default():
            restore_time = time.time()
            if self.args.get('--restore') is not None:
            #valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed = self.run_epoch("Test (validation)", self.valid_data, False)
            #with open('./outputs/test.log', 'a') as f:
                #print(line%(valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed), file = f)
                _, valid_accs, _, _, _, _ = self.run_epoch("Resumed (validation)", self.valid_data, False, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
                restore_time = time.time()
            else:
                (best_val_acc, best_val_acc_epoch) = (0., 0.)
            if is_test == False:
                for epoch in range(1, self.params['num_epochs'] + 1):
                    train_loss, train_accs, train_precision, train_recall, train_f1, train_speed = self.run_epoch("epoch %i (training)" % epoch, self.train_data, True, False)
                    valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed = self.run_epoch("epoch %i (validation)" % epoch, self.valid_data, False, False)
                    epoch_time = time.time() - total_time_start
                    print(summ_line%(epoch, self.params['train_file'], train_loss, train_accs, train_precision, train_recall, train_f1, train_speed, epoch_time))
                    print(summ_line%(epoch, self.params['valid_file'], valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed, epoch_time))
                #    with open('./outputs/train.log', 'a') as f:
                #        print(line%(train_loss, train_accs, train_precision, train_recall, train_f1, train_speed), file = f)
                #with open('./outputs/test.log', 'a') as f:
                #    print(line%(valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed), file = f)

                    bak_train_data.append([epoch, self.params['train_file'], train_loss, np.sum(train_accs), np.sum(train_precision), np.sum(train_recall), np.sum(train_f1), train_speed])
                    bak_valid_data.append([epoch, self.params['valid_file'], valid_loss, np.sum(valid_accs), np.sum(valid_precision), np.sum(valid_recall), np.sum(valid_f1), valid_speed])
                    if epoch == 1:
                        first_epoch = time.time()
                    if epoch == 2:
                        second_epoch = time.time()
                    if epoch == self.params['num_epochs']-1:
                        next2last_epoch = time.time()
                    if epoch == self.params['num_epochs']:
                        last_epoch = time.time()

                    if is_test == False:
                        val_acc = np.sum(valid_accs)  # type: float
                        if val_acc > best_val_acc:
                            self.save_model(self.best_model_file)
                            print("(Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (val_acc, best_val_acc, self.best_model_file))
                            best_val_acc = val_acc
                            best_val_acc_epoch = epoch
                    # elif epoch - best_val_acc_epoch >= self.params['patience']:
                    #     print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    #     break

                    if  self.params['timeout'] < epoch_time:
                        print("Stopping training after %i epochs timeout." % epoch)
                        break

        header = "epoch\tfile\tloss\taccs\tprecision\trecall\tf1\tspeed\n"
        if is_test == True:
            test_begin_time = time.time()
            batch_iterator = ThreadedIterator(self.make_minibatch_iterator(self.valid_data, False), max_queue_size=5)
            for batch_data in batch_iterator:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed = self.run_epoch("Test: ", batch_data, False, True)
                print("Test: %s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%(self.params['valid_file'], valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed))
            test_end_time = time.time()
            #with open(self.online_data_backup_file + "_train_final.txt", "w") as f:
            #    f.write(header)
            #    for line in bak_train_data:
            #        f.write("\t".join([str(item) for item in line]) + "\n")
                with open(self.online_data_backup_file + "_test.txt", "w") as f:
                    f.write("file\tloss\taccs\tprecision\trecall\tf1\tspeed\n")
                    f.write("\t".join([self.params['valid_file'], valid_loss, valid_accs, valid_precision, valid_recall, valid_f1, valid_speed]) + "\n")
        else :    
            with open(self.online_data_backup_file + "_train.txt", "w") as f:
                f.write(header)
                for line in bak_train_data:
                    f.write("\t".join([str(item) for item in line]) + "\n")
            with open(self.online_data_backup_file + "_valid.txt", "w") as f:
                f.write(header)
                for line in bak_valid_data:
                    f.write("\t".join([str(item) for item in line]) + "\n")
        
        with open('./outputs/time.log', 'a') as f:
            if not is_test:
                out_str = "init time:{}\trestore time:{}\tfirst train:{}\tlast train:{}".format(restore_time-train_begin_time, first_epoch-restore_time, second_epoch-first_epoch, last_epoch-next2last_epoch)
                f. write('---------- train: '+str(int(train_begin_time))+'----------\n')
            else:
                out_str = "init time:{}\trestore time:{}\ttest time:{}".format(restore_time-train_begin_time, test_begin_time-restore_time, test_end_time-test_begin_time)
                f. write('---------- test: '+str(int(train_begin_time))+'----------\n')
            f.write(out_str+'\n')
            print(out_str)
        
        
    def test(self):
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                _, valid_accs, _, _, _, _ = self.run_epoch("Test run", self.valid_data, False, True)
                best_val_acc = np.sum(valid_accs)
                print("Pred: %.2f" % best_val_acc)
                
    def generate_vector(self):
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                self.myalldata_dir = '/data/bwj/test/mangrove/ml/ggnn/data/singles/rw/'
                self.data_dir = self.myalldata_dir
                for _, _, files in os.walk(self.myalldata_dir):
                    for singlefile in files:
                        single_data = self.load_data(singlefile, is_training_data = False)
                        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(self.valid_data, False), max_queue_size=5)
                        for batch_data in batch_iterator:
                            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                            final_node = self.sess.run(self.compute_final_node_representations(),feed_dict=batch_data)
                            final_node = np.mean(final_node, 0)
                            np.save('./outputs/ggnn_vectors/{}_final_node.npy'.format(singlefile.replace('.json', '')), final_node)
                        #_, singel_accs, _, _, _, _ = self.run_epoch("Generate {}'s vector".format(singlefile), single_data, False, True)
                        #best_val_acc = np.sum(valid_accs)
                        #print("%s Pred: %.2f" % (singlefile, best_val_acc))
                        print("{} ok".format(singlefile))

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
                         "params": self.params,
                         "weights": weights_to_save
                       }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        #for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
            #if par not in ['task_ids', 'num_epochs', 'random_seed', 'train_file', 'valid_file']:
                #print(par, par_value, data_to_load['params'][par])
                #assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")
