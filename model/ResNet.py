from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, \
  Activation, GlobalAveragePooling2D


# 继承Layer,建立resnet50 101 152卷积层模块
def conv_block(inputs, filter_num, stride=1, name=None):
  x = inputs
  x = Conv2D(filter_num[0], (1, 1), strides=stride, padding='same', name=name + '_conv1')(x)
  x = BatchNormalization(axis=3, name=name + '_bn1')(x)
  x = Activation('relu', name=name + '_relu1')(x)

  x = Conv2D(filter_num[1], (3, 3), strides=1, padding='same', name=name + '_conv2')(x)
  x = BatchNormalization(axis=3, name=name + '_bn2')(x)
  x = Activation('relu', name=name + '_relu2')(x)

  x = Conv2D(filter_num[2], (1, 1), strides=1, padding='same', name=name + '_conv3')(x)
  x = BatchNormalization(axis=3, name=name + '_bn3')(x)

  # residual connection
  r = Conv2D(filter_num[2], (1, 1), strides=stride, padding='same', name=name + '_residual')(inputs)
  x = layers.add([x, r])
  x = Activation('relu', name=name + '_relu3')(x)

  return x


def build_block(x, filter_num, blocks, stride=1, name=None):
  x = conv_block(x, filter_num, stride, name=name)

  for i in range(1, blocks):
    x = conv_block(x, filter_num, stride=1, name=name + '_block' + str(i))

  return x


# 创建resnet50 101 152
def ResNet(Netname, nb_classes):
  ResNet_Config = {'ResNet50': [3, 4, 6, 3],
                   'ResNet101': [3, 4, 23, 3],
                   'ResNet152': [3, 8, 36, 3]}
  layers_dims = ResNet_Config[Netname]

  filter_block1 = [64, 64, 256]
  filter_block2 = [128, 128, 512]
  filter_block3 = [256, 256, 1024]
  filter_block4 = [512, 512, 2048]

  img_input = Input(shape=(224, 224, 3))
  # stem block
  x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='stem_conv')(img_input)
  x = BatchNormalization(axis=3, name='stem_bn')(x)
  x = Activation('relu', name='stem_relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='stem_pool')(x)
  # convolution block
  x = build_block(x, filter_block1, layers_dims[0], name='conv1')
  x = build_block(x, filter_block2, layers_dims[1], stride=2, name='conv2')
  x = build_block(x, filter_block3, layers_dims[2], stride=2, name='conv3')
  x = build_block(x, filter_block4, layers_dims[3], stride=2, name='conv4')
  # top layer
  x = GlobalAveragePooling2D(name='top_layer_pool')(x)
  x = Dense(nb_classes, activation='softmax', name='fc')(x)

  model = models.Model(img_input, x, name=Netname)

  return model


