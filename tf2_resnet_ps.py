import os
import ssl
import tensorflow as tf
import numpy as np
#### Necessary Imports for Neural Net
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow._api.v2.compat.v1 as tf1
from model.ResNet import ResNet


#TODO: 添加AlexNet, Inception, VGG 16的模型；iterator方式读入数据；验证集使用；checkpoint使用。
tf1.app.flags.DEFINE_string('ps_hosts', 'None', "private_ip1:port1, private_ip2:port2,....")
tf1.app.flags.DEFINE_string('worker_hosts', 'None', "")
tf1.app.flags.DEFINE_string('chief_host', 'None', "")
tf1.app.flags.DEFINE_string('task_name', 'None', "ps,worker,chief")
tf1.app.flags.DEFINE_integer('task_index', 0 , '')

FLAGS = tf1.app.flags.FLAGS

  ### Define some Callbacks
def lrdecay(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

def earlystop(mode):
  if mode=='acc':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, mode='max')
  elif mode=='loss':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
  return estop



if __name__ == '__main__':
    # Set the environment variable to allow reporting worker and ps failure to the
    # coordinator. This is a workaround and won't be necessary in the future.
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["GRPC_FAIL_FAST"] = "use_caller"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    cluster_dict = {}
    cluster_dict["worker"] = FLAGS.worker_hosts.split(',')
    cluster_dict["ps"] = FLAGS.ps_hosts.split(',')
    cluster_dict["chief"] = FLAGS.chief_host.split(',')
    cluster_spec = tf.train.ClusterSpec(cluster_dict)
    if FLAGS.task_name in ("worker", "ps"):
        # Set the environment variable to allow reporting worker and ps failure to the
        # coordinator. This is a workaround and won't be necessary in the future.
        os.environ["GRPC_FAIL_FAST"] = "use_caller"
        print(cluster_spec, FLAGS.task_name)
        server = tf.distribute.Server(
            cluster_spec,
            job_name=FLAGS.task_name,
            task_index=FLAGS.task_index,
            protocol="grpc",
            start=True)
        server.join()
    else:
        class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-100 website
        # Load Cifar-100 data-set
        (train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar100.load_data()
        #### Normalize the images to pixel values (0, 1)
        train_im, test_im = train_im/255.0 , test_im/255.0
        #### Check the format of the data 
        print ("train_im, train_lab types: ", type(train_im), type(train_lab))
        #### check the shape of the data
        print ("shape of images and labels array: ", train_im.shape, train_lab.shape) 
        print ("shape of images and labels array ; test: ", test_im.shape, test_lab.shape)

        #### Check the distribution of unique elements 
        (unique, counts) = np.unique(train_lab, return_counts=True)
        frequencies = np.asarray((unique, counts))
        print (frequencies)
        print (len(unique))

        ### One hot encoding for labels 

        train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=100, dtype='uint8')

        test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=100, dtype='uint8')

        ### Train -test split 
        train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical, test_size=0.20, 
                                                                    stratify=train_lab_categorical, 
                                                                    random_state=40, shuffle = True)

        print ("train data shape after the split: ", train_im.shape)
        print ('new validation data shape: ', valid_im.shape)
        print ("validation labels shape: ", valid_lab.shape)


        # train_dateset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
        print("############## Step: dataset prepared, set lrdecay next...")
        lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay 
        print("############## Step: set lrdecay, partition variable next...")
        variable_partitioner = (
            tf.distribute.experimental.partitioners.MinSizePartitioner(
                min_shard_bytes=(256 << 10),
                max_shards=len(FLAGS.ps_hosts.split(','))))
        print("############## Step: config cluster reslover next...")
        cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, rpc_layer="grpc")
        print("############## Step: define ParameterServerStrategy next...")
        strategy = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver,
            variable_partitioner=variable_partitioner)


        # print("############## Step: set coordinator next...")   
        # coordinator =tf.distribute.experimental.coordinator.ClusterCoordinator(strategy=strategy)
        # print("############## Step: prepare distributed dataset next...")  
        # distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
        print("############## Step: model definition...")
        with strategy.scope():
            resnet50_model = ResNet('ResNet50', 100)
        # from tensorflow.python.client import device_lib
        # from tensorflow.keras.utils import multi_gpu_model
        # gpus_num = len([ 1 for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
        # if gpus_num == 0:
        #     para_model = resnet50_model
        # elif gpus_num == 1:
        #     with tf.device("/gpu:0"):
        #         para_model = resnet50_model
        # else:
        #     para_model=multi_gpu_model(resnet50_model, gpus=gpus_num)

        resnet50_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                            metrics=['acc'])

    #     working_dir="/tmp/tf2_result/"
    #     log_dir = os.path.join(working_dir, 'log')
    #     ckpt_filepath = os.path.join(working_dir, 'ckpt')
    #     backup_dir = os.path.join(working_dir, 'backup')

        callbacks = [
        lrdecay
    #     ,tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    #     ,tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath)
    #     ,tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir)
        ]
        print("############## Step: training begins...")
        def dataset_fn(input_context):
            global_batch_size = 64
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            dataset = tf.data.Dataset.from_tensor_slices((train_im, train_lab)).shuffle(64).repeat()
            dataset = dataset.shard(
                input_context.num_input_pipelines,
                input_context.input_pipeline_id)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(2)
            return dataset


        distributed_dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
        resnet_train = resnet50_model.fit(distributed_dataset,
                                        epochs=100, 
                                        steps_per_epoch=train_im.shape[0]/400, 
    #                                     validation_steps=valid_im.shape[0]/batch_size, 
    #                                     validation_data=validation_dataset, 
                                        callbacks=callbacks)
