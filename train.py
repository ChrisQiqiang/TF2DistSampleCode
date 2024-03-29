import os
import ssl
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow._api.v2.compat.v1 as tf1
from model.VGG import vgg_16

from datetime import datetime

tf1.app.flags.DEFINE_string('ps_hosts', 'None', "private_ip1:port1, private_ip2:port2,....")
tf1.app.flags.DEFINE_string('worker_hosts', 'None', "")
tf1.app.flags.DEFINE_string('chief_host', 'None', "")
tf1.app.flags.DEFINE_string('task_name', 'None', "ps,worker,chief")
tf1.app.flags.DEFINE_integer('task_index', 0 , '')
tf1.app.flags.DEFINE_string('model_name', 'inception', '')
tf1.app.flags.DEFINE_integer('batch_size', 128 , '')
tf1.app.flags.DEFINE_string('dataset', 'cifar100' , '')
tf1.app.flags.DEFINE_integer('profiler_enable', 0, '')

FLAGS = tf1.app.flags.FLAGS

### Define some Callbacks
# def lrdecay(epoch):
#     lr = 1e-2
#     if epoch > 160:
#         lr *= 0.5e-3
#     elif epoch > 120:
#         lr *= 1e-3
#     elif epoch > 80:
#         lr *= 1e-2
#     elif epoch > 20:
#         lr *= 1e-1
#     return lr

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
    model_name=FLAGS.model_name
    if FLAGS.task_name in ("worker", "ps"):
        # Set the environment variable to allow reporting worker and ps failure to the
        # coordinator. This is a workaround and won't be necessary in the future.
        print(cluster_spec, FLAGS.task_name)
        server = tf.distribute.Server(
            cluster_spec,
            job_name=FLAGS.task_name,
            task_index=FLAGS.task_index,
            protocol="grpc",
            start=True)
        server.join()
    else:
        # Load Cifar-10 or Cifar-100 dataset
        num_class = 0
        assert FLAGS.dataset in ('cifar10','cifar100')
        (train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar10.load_data() if FLAGS.dataset == 'cifar10' else  tf.keras.datasets.cifar100.load_data()
        num_class = 10 if FLAGS.dataset == 'cifar10' else 100

        #### Normalize the images to pixel values (0, 1)
        train_im, test_im = train_im / 255.0, test_im / 255.0
        #### Check the format of the data
        print("train_im, train_lab types: ", type(train_im), type(train_lab))
        #### check the shape of the data
        print("shape of images and labels array: ", train_im.shape, train_lab.shape)
        print("shape of images and labels array ; test: ", test_im.shape, test_lab.shape)

        #### Check the distribution of unique elements
        (unique, counts) = np.unique(train_lab, return_counts=True)
        frequencies = np.asarray((unique, counts))
        print(frequencies)
        print(len(unique))

        ### One hot encoding for labels

        train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=num_class, dtype='uint8')
        test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=num_class, dtype='uint8')


        ### Train -test split
        train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical, test_size=0.1,
                                                                    stratify=train_lab_categorical,
                                                                    random_state=40, shuffle = True)

#         train_im = tf.image.resize_with_pad(train_im, target_height=224, target_width=224)
#         print("train data shape after the split: ", train_im.shape)
#         print('new validation data shape: ', valid_im.shape)
#         print("validation labels shape: ", valid_lab.shape)


        # train_dateset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
        print("############## Step: dataset prepared, set lrdecay next...")
#         lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay
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

        print("############## Step: model definition...")
        with strategy.scope():
            if model_name.lower() == "inception":
                from tensorflow.keras.applications.inception_v3 import InceptionV3
                model=InceptionV3(weights=None, classes=num_class)
            elif model_name.lower() == "vgg16":
                model=vgg_16()
            # elif model_name.lower() == "vgg19":
            #     model=vgg_19()
            elif model_name.lower() == "resnet152":
                from tensorflow.keras.applications.resnet import ResNet152
                model = ResNet152(weights=None, classes=num_class)
            elif model_name.lower() == "resnet50":
                from tensorflow.keras.applications.resnet import ResNet50
                model = ResNet50(weights=None, classes=num_class)
            elif model_name.lower() == "resnet101":
                from tensorflow.keras.applications.resnet import ResNet101
                model = ResNet101(weights=None, classes=num_class)
            else:
                ex = Exception("Exception: your model is not supported by our python script, please build your model by yourself.")
                raise ex
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-2),metrics=['accuracy', 'top_k_categorical_accuracy'])
            
        print("##############配置训练相关参数，图片预处理，callback函数等...")
    #     ckpt_filepath = os.path.join("/tmp/tf2_result/", 'ckpt')
    #     backup_dir = os.path.join("/tmp/tf2_result/", 'backup')
        log_dir = "/code/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#         callbacks = [
#          lrdecay
    #     ,tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath)
    #     ,tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir)
#         ]
        if FLAGS.profiler_enable:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=1))

        global_batch_size = FLAGS.batch_size


        def dataset_fn(input_context):
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            print("####### train model :", FLAGS.model_name)
            print("####### dataset :", FLAGS.dataset)
            print("####### batch size :", batch_size)
            dataset = tf.data.Dataset.from_tensor_slices((train_im, train_lab)).shuffle(64).batch(batch_size).repeat(500) \
                .map(lambda x, y: (tf.image.resize(x,[299,299]), y))
            dataset = dataset.prefetch(1)
            return dataset

        distributed_dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

        def valid_dataset_fn(input_context):
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            dataset = tf.data.Dataset.from_tensor_slices((valid_im, valid_lab)).shuffle(64).batch(batch_size).repeat(500) \
                .map(lambda x, y: (tf.image.resize_with_crop_or_pad(x, target_height=299, target_width=299), y))
            dataset = dataset.prefetch(1)
            return dataset

        validation_dataset = tf.keras.utils.experimental.DatasetCreator(valid_dataset_fn)

        # iteration number = epochs * steps_per_epoch
        # steps_per_epoch为一个epoch里有多少次batch迭代（即一个epoch里有多少个iteration）
        print("############## Step: training begins...")
        model.fit(distributed_dataset, epochs=800, steps_per_epoch=30
#                                         ,validation_steps=valid_im.shape[0]/global_batch_size
#                                         ,validation_data=validation_dataset
#                                        , callbacks=callbacks
                                        )
