import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import os

# number of features in the criteo dataset after one-hot encoding
num_features = 33762578 # DO NOT CHANGE THIS VALUE

s_batch = 100
eta = 1
train_test_ratio = 2000
total_trains = 200001
iterations = total_trains/(5*s_batch)


s_test = 20;
total_tests = 20000;

file_dict = {0:["00","01","02","03","04"],
             1:["05","06","07","08","09"],
             2:["10","11","12","13","14"],
             3:["15","16","17","18","19"],
             4:["20","21"],
            -1:["22"]}

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

def print_specs():
    print "====================Inforamtion======================="
    print "task ID------------------------------------------",FLAGS.task_index
    print "test_train_ratio:--------------------------------",train_test_ratio
    print "training batch size per iteration:---------------", s_batch
    print "testing batch size per iteration:----------------", s_test
    print "total size of test set:--------------------------",total_tests
    print "total training iterations:-----------------------",iterations
    print "# training iterations before each testing period:",( train_test_ratio/(5*s_batch) )
    print "# of iterations per testing period:--------------", total_tests/s_test
    print "======================================================"






g = tf.Graph()

with g.as_default():


    def get_datapoint_iter(file_idx=[],batch_size=s_batch):
        fileNames = map(lambda s: "/home/ubuntu/criteo-tfr-tiny/tfrecords"+s,file_idx)
        # We first define a filename queue comprising 5 files.
        filename_queue = tf.train.string_input_producer(fileNames, num_epochs=None)


        # TFRecordReader creates an operator in the graph that reads data from queue
        reader = tf.TFRecordReader()

        # Include a read operator with the filenae queue to use. The output is a string
        # Tensor called serialized_example
        _, serialized_example = reader.read(filename_queue)


        # The string tensors is essentially a Protobuf serialized string. With the
        # following fields: label, index, value. We provide the protobuf fields we are
        # interested in to parse the data. Note, feature here is a dict of tensors
        features = tf.parse_single_example(serialized_example,
                                           features={
                                            'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                            'index' : tf.VarLenFeature(dtype=tf.int64),
                                            'value' : tf.VarLenFeature(dtype=tf.float32),
                                           }
                                          )

        label = features['label']
        index = features['index']
        value = features['value']


        # combine the two sparseTensors into one that will be used in gradient calcaulation
        index_shaped = tf.reshape(index.values,[-1, 1])
        value_shaped = tf.reshape(value.values,[-1])
        combined_values = tf.sparse_transpose( 
                            tf.SparseTensor(indices=index_shaped,
                                        values=value_shaped,
                                        shape=[33762578]) 
                            )


        label_flt = tf.cast(label, tf.float32)
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        value_batch,label_batch = tf.train.shuffle_batch(
          [combined_values, label_flt], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)

        return value_batch,label_batch
    #############################################################
    
    def calc_gradient(X,W,Y):
        pred = tf.mul(tf.sparse_tensor_dense_matmul(X, W), -1.0)
        error = tf.sigmoid(tf.mul(Y,pred))
        error_m1 = error-1

        error_Y = tf.mul(Y,error_m1)
        X_T = tf.sparse_transpose(X)

        gradient = tf.sparse_tensor_dense_matmul(X_T,error_Y)
        return tf.reshape(gradient, [num_features,1])
    #############################################################

    # creating a model variable on task 0. This is a process running on node vm-48-1
    with tf.device("/job:worker/task:0"):
        w_a = tf.Variable(tf.ones([num_features, 1])*.1, name="model_asynch")

    # creating only reader and gradient computation operator
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        # read the data
        X_a,Y_a = get_datapoint_iter(
                        file_dict[FLAGS.task_index],
                        batch_size=s_batch
                    )
        # calculate the gradient
        local_gradient_a = calc_gradient(X_a,w_a,Y_a)
        # multiple the gradient with the learning rate and submit it to update the model
        gradients_a = tf.mul(local_gradient_a, eta)


    with tf.device("/job:worker/task:0"):
        agg_shape_a = tf.reshape(gradients_a,[num_features, 1])

        update_log_a = tf.matmul(tf.transpose(agg_shape_a),agg_shape_a )
        assign_op_a = w_a.assign_add(agg_shape_a)
    ###########################################################
    ###########################################################
    def calc_precision(W,X,Y):
        pred = tf.sparse_tensor_dense_matmul(X, W)
        diffs = tf.sign(tf.mul(Y,pred))

        precision = tf.reduce_sum((diffs+1)/2)/s_test
        return precision

    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        test_X,test_Y = get_datapoint_iter(file_dict[-1],batch_size = s_test)
        precision = calc_precision(w_a,test_X,test_Y)

    ###########################################################
    with tf.Session("grpc://vm-32-%d:2222" % (FLAGS.task_index+1)) as sess_asynch:
        print_specs()
        # only one client initializes the variable
        if FLAGS.task_index == 0:
            sess_asynch.run(tf.initialize_all_variables())

        ###################################################################
        # utility function to report the precision during training 
        def report_precision():
            print "------------reporting precision------------"
            # with tf.device("/job:worker/task:0"):
            #     test_X,test_Y = get_datapoint_iter(file_dict[-1],batch_size = s_test)
            #     precision = calc_precision(w,test_X,test_Y)

            out_prec = []
            for j in range(total_tests/s_test):
                out_prec.append(precision.eval())
                #print "precision: ",out_prec[j]
            #print "precision vector:",out_prec
            print "total precision:", np.mean(out_prec), "max:", np.max(out_prec)
        ###################################################################

        # this is new command and is used to initialize the queue based readers.
        # Effectively, it spins up separate threads to read from the files
        coord_a = tf.train.Coordinator()
        threads_a = tf.train.start_queue_runners(sess=sess_asynch, coord=coord_a)
        ###################################################################
        for i in range(iterations):
            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            options_a = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            
            run_metadata_a = tf.RunMetadata()
            print "Step ",i
            sess_asynch.run(assign_op_a)
            print "update log:",update_log_a.eval()
            # sess.run(assign_op,options=options, run_metadata=run_metadata)
            # print "ulog ", ulog
            # start testing period
            if i%( train_test_ratio/(5*s_batch) ) == 0:
                report_precision();


        # Create the Timeline object, and write it to a json file
        # fetched_timeline = timeline.Timeline(
        #                         run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('timeline_01.json', 'w') as f:
        #     f.write(chrome_trace)

        coord_a.request_stop()
        coord_a.join(threads_a, stop_grace_period_secs=5)
        sess_asynch.close()
