import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import os

# number of features in the criteo dataset after one-hot encoding
num_features = 33762578 # DO NOT CHANGE THIS VALUE

s_batch = 100
eta = 0.2
train_test_ratio = 20000
total_trains = 200001
iterations = total_trains/(5*s_batch)


s_test = 20;
total_tests = 2000;

file_dict = {0:["00","01","02","03","04"],
             1:["05","06","07","08","09"],
             2:["10","11","12","13","14"],
             3:["15","16","17","18","19"],
             4:["20","21"],
            -1:["22"]}


def print_specs():
    print "====================Inforamtion======================="
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

    def calc_gradient(X,W,Y):
        pred = tf.mul(tf.sparse_tensor_dense_matmul(X, W), -1.0)
        error = tf.sigmoid(tf.mul(Y,pred))
        error_m1 = error-1

        error_Y = tf.mul(Y,error_m1)
        X_T = tf.sparse_transpose(X)


        gradient_before = tf.sparse_tensor_dense_matmul(X_T,error_Y)
        gradient = tf.reshape(gradient_before, [num_features,1])
        return gradient

    # creating a model variable on task 0. This is a process running on node vm-32-1
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([num_features, 1])*.1, name="model")

    # creating 5 reader operators to be placed on different operators
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    gradients = []
    for i in range(0, 5):
        with tf.device("/job:worker/task:%d" % i):
            # read the data
            X,Y = get_datapoint_iter(file_dict[i],batch_size=s_batch)
            # calculate the gradient
            local_gradient = calc_gradient(X,w,Y)
            # multiple the gradient with the learning rate and submit it to update the model
            gradients.append(tf.mul(local_gradient, eta))


    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        aggregator = tf.add_n(gradients)
        agg_shape = tf.reshape(aggregator,[num_features, 1])
      
        update_log = tf.reduce_mean(tf.abs(agg_shape))
        #update_log_before = tf.matmul(tf.transpose(tf.div(agg_shape,5.0*s_batch)),tf.div(agg_shape,5.0*s_batch))
        #update_log = tf.reshape(update_log_before,[1,1])        
        update_log = tf.matmul(tf.transpose(agg_shape),agg_shape)
        assign_op = w.assign_add(agg_shape)

    ###########################################################
    def calc_precision(W,X,Y):
        pred = tf.sparse_tensor_dense_matmul(X, W)
        diffs = tf.sign(tf.mul(Y,pred))

        precision = tf.reduce_sum((diffs+1)/2)/s_test
        return precision

    with tf.device("/job:worker/task:0"):
        test_X,test_Y = get_datapoint_iter(file_dict[-1],batch_size = s_test)
        precision = calc_precision(w,test_X,test_Y)

    ###########################################################
    with tf.Session("grpc://vm-32-1:2222") as sess:
        # print the model specification to terminal 
        print_specs()

        sess.run(tf.initialize_all_variables())
        
        # tf.train.start_queue_runners(sess=sess)

        

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
            # print "precision vector:",out_prec
            print "total precision:", np.mean(out_prec), "max:", np.max(out_prec)
        ###################################################################

        # this is new command and is used to initialize the queue based readers.
        # Effectively, it spins up separate threads to read from the files
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # main loop of training
        for i in range(iterations):
            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            
            run_metadata = tf.RunMetadata()
            print "Step ",i
            sess.run(assign_op)
            print "update log:",update_log.eval()
            # sess.run(assign_op,options=options, run_metadata=run_metadata)
            # print "ulog ", ulog

            # start testing period
            if i%( train_test_ratio/(5*s_batch) ) == 0:
                report_precision();

            # print w.eval()
        # Create the Timeline object, and write it to a json file
        # fetched_timeline = timeline.Timeline(
        #                         run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('timeline_01.json', 'w') as f:
        #     f.write(chrome_trace)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=5)
        tf.train.SummaryWriter("%s/synchSGD" % (os.environ.get("TF_LOG_DIR")), sess.graph)
        sess.close()
