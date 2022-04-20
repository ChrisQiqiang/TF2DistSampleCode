import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

from VGG import vgg_16
from ResNet import ResNet
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def loadDataset(dataset_folder):  # 训练集
    dataset = np.zeros((10000 * 5, 3 * 32 * 32), dtype=np.int32)  # 训练集 先用0填充,每个元素都是4byte integer
    labels = np.zeros((10000 * 5), dtype=np.int32)
    for i in range(5):
        d = unpickle(os.path.join(dataset_folder, "data_batch_%d" % (i + 1)))
        # 每个文件含1万张图片的数据
        for j in range(len(d[b'labels'])):  # 每张图片，shape: (3072,)
            dataset[10000 * i + j] = d[b'data'][j]
            labels[10000 * i + j] = d[b'labels'][j]
    reshaped = np.reshape(dataset, (10000 * 5, 3, 32, 32))
    # 交换轴
    swapaxesed = np.swapaxes(reshaped, 1, 2)  # 按图片，行，颜色通道 ，列 排列
    swapaxesed = np.swapaxes(swapaxesed, 2, 3)  # 按图片，行，列，颜色通道 排列

    return swapaxesed, labels


def loadValidset(path):  # 验证集
    valid_dataset = np.zeros((10000, 3 * 32 * 32), dtype=np.int32)  # 训练集 先用0填充,每个元素都是4byte integer
    valid_labels = np.zeros((10000), dtype=np.int32)
    d = unpickle(path)
    # 测试含1万张图片的数据
    for j in range(len(d[b'labels'])):  # 每张图片，shape: (3072,)
        valid_dataset[j] = d[b'data'][j]
        valid_labels[j] = d[b'labels'][j]
    reshaped = np.reshape(valid_dataset, (10000, 3, 32, 32))
    # 交换轴
    swapaxesed = np.swapaxes(reshaped, 1, 2)  # 按图片，行，颜色通道 ，列 排列
    swapaxesed = np.swapaxes(swapaxesed, 2, 3)  # 按图片，行，列，颜色通道 排列
    return swapaxesed, valid_labels
#
def local_cifar10_test(dataset_folder):
    # 测试数据集是否加载成功
    train_dataset, train_labels = loadDataset(dataset_folder)
    valid_dataset, valid_labels = loadValidset(os.path.join(dataset_folder, "test_batch"))
    print("Train data:", np.shape(train_dataset), np.shape(train_labels))
    print("Test data :", np.shape(valid_dataset), np.shape(valid_labels))
    # 图片识别时才需要
    label_names = unpickle(os.path.join(dataset_folder, "batches.meta"))
    names = label_names[b'label_names']  # 分类结果字节字符串
    model_vgg = vgg_16()
    train_labels = tf.one_hot(train_labels, 10)
    model_vgg.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['acc'])
    # model_vgg.summary()
    model_vgg.fit(x=train_dataset, y=train_labels, epochs=100, batch_size=10)
    # model_vgg.summary()
    ###显示图片
    # print(min(train_labels))  # 标签编码从0开始
    # for index in range(100, 200):
    #     a = valid_dataset[index]
    #     print(a.shape)
    #     print(a)
    #     plt.imshow(a)
    #     print(valid_labels[index])
    #     plt.title(names[int(valid_labels[index])], fontsize=8)
    #     plt.xticks([]);
    #     plt.yticks([])
    #     plt.show()



def random_inference():
    ###创建一个3*32*32的图片，所有取值在0-256, 取值需要转为float
    x = [ random.randint(0 ,256) * 1.0 for i in range( 3* 32* 32)]
    input_tensor = tf.constant(x, shape=[1 ,3 ,32 ,32])
    model_vgg = vgg_16()
    print(type(model_vgg))
    with tf.GradientTape() as tape:
        predictions = model_vgg(input_tensor)
        print(predictions)
        test = tf.argmax(predictions,axis=1)
        print(test)


def keras_cifar10_test():
    num_class=10
    (train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar10.load_data()
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
    train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical, test_size=0.90,
                                                                stratify=train_lab_categorical,
                                                                random_state=40, shuffle=True)
    train_im = tf.image.resize_with_pad(train_im, target_height=224, target_width=224)
    print("train data shape after the split: ", train_im.shape)
    print('new validation data shape: ', valid_im.shape)
    print("validation labels shape: ", valid_lab.shape)

    model = vgg_16()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                  metrics=['acc'])
    model.summary()
    model.fit(train_im, train_lab, epochs=10, batch_size=32)

if __name__ == '__main__':
    #本地cifar10训练，路径写明下载解压的cifar10数据集位置即可。
    # dataset_folder = r"..\dataset\cifar-10-batches-py"
    # local_cifar10_test(dataset_folder)
    #推理测试
    # random_inference()
    # #keras load cifar10 进行训练
    keras_cifar10_test()