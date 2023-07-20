import cv2
from tensorflow.keras.utils import img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, MaxPool2D, concatenate,Layer,BatchNormalization, Activation
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from glob import glob
from tqdm import tqdm
import time

SIZE = 160


# Encoder Layer Block
class EncoderLayerBlockLinknet(Layer):
  def __init__(self, filters, rate, pooling=True):
    super(EncoderLayerBlockLinknet, self).__init__()
    self.filters = filters
    self.rate = rate
    self.pooling = pooling

    self.c1 = Conv2D(self.filters, kernel_size=3, padding='same', kernel_initializer='he_normal')
    self.bn1 = BatchNormalization()
    self.a1 = Activation('relu')
    self.c2 = Conv2D(self.filters, kernel_size=3, padding='same', kernel_initializer='he_normal')
    self.bn2 = BatchNormalization()
    self.a2 = Activation('relu')
    self.pool = MaxPool2D(pool_size=(2,2))

  def call(self, X):
    x = self.c1(X)
    x = self.bn1(x)
    x = self.a1(x)
    x = self.c2(x)
    x = self.bn2(x)
    x = self.a2(x)
    if self.pooling:
      y = self.pool(x)
      return y, x
    else:
      return x

  def get_config(self):
    base_estimator = super().get_config()
    return {
        **base_estimator,
        "filters":self.filters,
        "rate":self.rate,
        "pooling":self.pooling
    }

# Decoder Layer Block
class DecoderLayerBlockLinknet(Layer):
  def __init__(self, filters, rate, padding='same'):
    super(DecoderLayerBlockLinknet, self).__init__()
    self.filters = filters
    self.rate = rate
    self.cT = Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding=padding)
    self.bnT = BatchNormalization()
    self.aT = Activation('relu')
    self.next = EncoderLayerBlockLinknet(self.filters, self.rate, pooling=False)

  def call(self, X):
    X, skip_X = X
    x = self.cT(X)
    x = self.bnT(x)
    x = self.aT(x)
    c1 = concatenate([x, skip_X])
    y = self.next(c1)
    return y

  def get_config(self):
    base_estimator = super().get_config()
    return {
        **base_estimator,
        "filters":self.filters,
        "rate":self.rate,
    }

input_layer_linknet = tf.keras.Input([160,160,3])

# Encoder
p1, c1 = EncoderLayerBlockLinknet(16, 0.1)(input_layer_linknet)
p2, c2 = EncoderLayerBlockLinknet(32, 0.1)(p1)
p3, c3 = EncoderLayerBlockLinknet(64, 0.2)(p2)
p4, c4 = EncoderLayerBlockLinknet(128, 0.2)(p3)

# Decoder
d1 = DecoderLayerBlockLinknet(128, 0.2)([p4, c4])
d2 = DecoderLayerBlockLinknet(64, 0.2)([d1, c3])
d3 = DecoderLayerBlockLinknet(32, 0.2)([d2, c2])
d4 = DecoderLayerBlockLinknet(16, 0.2)([d3, c1])

# Output layer
outputlinknet = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid')(d4)

# LinkNet Model
LinknetModel = tf.keras.models.Model(
    inputs=[input_layer_linknet],
    outputs=[outputlinknet],
)

LinknetModel.load_weights(r'D:\AI_VSCode\pytorch_unet\LinkNet.h5')


# Encoder Layer Block
class EncoderLayerBlockUnet(Layer):
  def __init__(self, filters, rate, pooling=True):
    super(EncoderLayerBlockUnet, self).__init__()
    self.filters = filters
    self.rate = rate
    self.pooling = pooling

    self.c1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
    self.drop = Dropout(self.rate)
    self.c2 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
    self.pool = MaxPool2D(pool_size=(2,2))

  def call(self, X):
    x = self.c1(X)
    x = self.drop(x)
    x = self.c2(x)
    if self.pooling:
      y = self.pool(x)
      return y, x
    else:
      return x

  def get_config(self):
    base_estimator = super().get_config()
    return {
        **base_estimator,
        "filters":self.filters,
        "rate":self.rate,
        "pooling":self.pooling
    }

#  Decoder Layer
class DecoderLayerBlockUnet(Layer):
  def __init__(self, filters, rate, padding='same'):
    super(DecoderLayerBlockUnet, self).__init__()
    self.filters = filters
    self.rate = rate
    self.cT = Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding=padding)
    self.next = EncoderLayerBlockUnet(self.filters, self.rate, pooling=False)

  def call(self, X):
    X, skip_X = X
    x = self.cT(X)
    c1 = concatenate([x, skip_X])
    y = self.next(c1)
    return y

  def get_config(self):
    base_estimator = super().get_config()
    return {
        **base_estimator,
        "filters":self.filters,
        "rate":self.rate,
    }

input_unet = tf.keras.Input(shape=(160,160,3))
p1, c1 = EncoderLayerBlockUnet(16,0.1)(input_unet)
p2, c2 = EncoderLayerBlockUnet(32,0.1)(p1)
p3, c3 = EncoderLayerBlockUnet(64,0.2)(p2)
p4, c4 = EncoderLayerBlockUnet(128,0.2)(p3)

c5 = EncoderLayerBlockUnet(256,0.3,pooling=False)(p4)

# Decoder
d1 = DecoderLayerBlockUnet(128,0.2)([c5, c4])
d2 = DecoderLayerBlockUnet(64,0.2)([d1, c3])
d3 = DecoderLayerBlockUnet(32,0.2)([d2, c2])
d4 = DecoderLayerBlockUnet(16,0.2)([d3, c1])

# Output layer
output_unet = Conv2D(3,kernel_size=1,strides=1,padding='same',activation='sigmoid')(d4)

Unetmodel = tf.keras.models.Model(
        inputs=[input_unet],
        outputs=[output_unet],
    )

Unetmodel.load_weights(r'D:\AI_VSCode\pytorch_unet\UNet.h5')

video_path = "D:\AI_VSCode\pytorch_unet\Vid\cityscapes_test2.mp4"  
# Đọc video
video = cv2.VideoCapture(video_path)
SIZE = 160
frame_count = 0
start_time = time.time()

# Lặp qua các khung hình trong video
cv2.namedWindow('unet', cv2.WINDOW_NORMAL)
cv2.namedWindow('linknet', cv2.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (160, 160))
    if not ret:
      break
    elapsed_time = time.time() - start_time
    frame_count += 1
    fps = frame_count / elapsed_time
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = img_to_array(image).astype('float')
    img = image[:,:256,:]/255.0
    # img = tf.image.resize(img,(SIZE, SIZE))
    img = img[np.newaxis,...]
    Unet_pred = Unetmodel.predict(img)[0]
    Unet_pred = cv2.resize(Unet_pred,(800,800))

    Linknet_pred = LinknetModel.predict(img)[0]
    Linknet_pred = cv2.resize(Linknet_pred,(800,800))

    frame = cv2.resize(frame, (800, 800))

    cv2.putText(Linknet_pred, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('LinkNet', Linknet_pred)

    cv2.putText(Unet_pred, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('UNet', Unet_pred)
    

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
