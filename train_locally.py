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
tf1.app.flags.DEFINE_string('model_name', 'alexnet', '')
tf1.app.flags.DEFINE_integer('batch_size', 512 , '')
tf1.app.flags.DEFINE_string('dataset', 'cifar100' , '')

FLAGS = tf1.app.flags.FLAGS

  ### Define some Callbacks
def lrdecay(epoch):
    lr = 1e-2
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
    num_class = 100
    (train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar100.load_data()

    #### Normalize the images to pixel values (0, 1)
    train_im, test_im = train_im / 255.0, test_im / 255.0

    #### Check the distribution of unique elements
    # (unique, counts) = np.unique(train_lab, return_counts=True)
    # frequencies = np.asarray((unique, counts))
    # print(frequencies)
    # print(len(unique))

    # train_in = [tf.image.resize_with_pad(x, target_height=224, target_width=224) for x in train_im]
    # train_im = train_in
    #### Check the format of the data
    print("train_im, train_lab types: ", type(train_im), type(train_lab))
    #### check the shape of the data
    print("shape of images and labels array: ", train_im.shape, train_lab.shape)
    print("shape of images and labels array ; test: ", test_im.shape, test_lab.shape)

    ### One hot encoding for labels

    train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=num_class, dtype='uint8')
    test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=num_class, dtype='uint8')

    ### Train -test split
    train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical, test_size=0.1,
                                                                stratify=train_lab_categorical,
                                                                random_state=40, shuffle=True)

    # dataset = tf.data.Dataset.from_tensor_slices((train_im, train_lab)).shuffle(64)
    # print(dataset.element_spec)
    # dataset2 = dataset.map(lambda x, y: (tf.image.resize_with_pad(x, target_height=224, target_width=224), y))
    # print(dataset2.element_spec)
    # dataset1 = dataset2.batch(16)
    # print(dataset1.element_spec)

    def dataset_fn(input_context):
        dataset = tf.data.Dataset.from_tensor_slices((train_im, train_lab)).shuffle(64).batch(16) \
                .map(lambda x, y: (tf.image.resize_with_crop_or_pad(x, target_height=224, target_width=224), y))
        dataset = dataset.prefetch(1)
        return dataset
    distributed_dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

    # from tensorflow.keras.applications.resnet import ResNet50
    # model = ResNet50(weights=None, classes=num_class)
    from model.VGG import vgg_16
    model = vgg_16(100)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                  metrics=['acc'])
    # log_dir = "../logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)

    model.fit(distributed_dataset, epochs=10, batch_size=2,
              # validation_steps=valid_im.shape[0]/32,
              # validation_data=validation_dataset,
              steps_per_epoch=100
              # ,callbacks=[tensorboard_callback]
              )
