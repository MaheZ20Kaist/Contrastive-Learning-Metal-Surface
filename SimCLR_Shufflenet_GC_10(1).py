




# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers
import math
import keras

# %%
!pip install tensorflow-addons

# %%
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D,LayerNormalization
from keras.layers import AveragePooling2D, Concatenate
from keras.layers import GlobalAveragePooling2D, Lambda
from keras.layers import BatchNormalization, DepthwiseConv2D
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_source_inputs
from keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
import tensorflow_addons as tfa



# %%
os.chdir("pretraining_new/")

# %%
!rm *.jpg~

# %%
unlabled_img_names = os.listdir()
unlabled_img_labels = [-1]*len(unlabled_img_names)
unlabled_img_names = [os.path.join("pretraining_new", path) for path in unlabled_img_names]

# %%
BATCH_SIZE = 32
IMAGE_SIZE = (200, 200)
UNLABLED_DATA_SIZE = 21222
LABLED_DATA_SIZE_TRAIN = 1260
LABLED_DATA_SIZE_TEST = 270
LABLED_DATA_SIZE_VAL = 270

STEPS_PER_EPOCH = (UNLABLED_DATA_SIZE + LABLED_DATA_SIZE_TRAIN) // BATCH_SIZE
UNLABELED_BATCH_SIZE = UNLABLED_DATA_SIZE // STEPS_PER_EPOCH
LABLED_BATCH_SIZE_TRAIN = LABLED_DATA_SIZE_TRAIN // STEPS_PER_EPOCH
LABLED_BATCH_SIZE_TEST = LABLED_DATA_SIZE_TEST//(STEPS_PER_EPOCH//10)
LABLED_BATCH_SIZE_VAL = LABLED_DATA_SIZE_VAL//(STEPS_PER_EPOCH//10)


print(STEPS_PER_EPOCH, UNLABELED_BATCH_SIZE, LABLED_BATCH_SIZE_TRAIN,LABLED_BATCH_SIZE_TEST,LABLED_BATCH_SIZE_VAL)

AUTOTUNE = tf.data.AUTOTUNE

# %%
def decode_jpg(img_name):
    img = tf.io.read_file(img_name)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def preprocessing_jpg(img_name, unlabled_img_labels):
    return decode_jpg(img_name), unlabled_img_labels



# %%
os.chdir("../")
os.chdir("neu-metal-surface/train/")

# %%
import glob
"""
Cr - 0
Sc - 1
In - 2
PS - 3
RS - 4
Pa - 5
"""
ls_Cr = glob.glob("Cr*.jpg")
ls_Sc = glob.glob("Sc*.jpg")
ls_In = glob.glob("In*.jpg")
ls_PS = glob.glob("PS*.jpg")
ls_RS = glob.glob("RS*.jpg")
ls_Pa = glob.glob("Pa*.jpg")

# %%
train_image_path, train_image_labels = ls_Cr + ls_Sc + ls_In + ls_PS + ls_RS + ls_Pa, [0]*len(ls_Cr) + [1]*len(ls_Sc) + [2]*len(ls_In) + [3]*len(ls_PS) + [4]*len(ls_RS) + [5]*len(ls_Pa)

# %%
train_image_path = [os.path.join("neu-metal-surface", "train", path) for path in train_image_path]
len(train_image_labels) == len(train_image_path)

# %%
def decode_bmp(img_name):
    img = tf.io.read_file(img_name)
    img = tf.image.decode_bmp(img, channels = 3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def preprocessing_bmp(img_name, labels):
    return decode_bmp(img_name), labels

# %%
unlabled_data = tf.data.Dataset.from_tensor_slices((unlabled_img_names, unlabled_img_labels))
unlabled_data = unlabled_data.map(preprocessing_jpg)
unlabled_data = unlabled_data.shuffle(UNLABLED_DATA_SIZE)
unlabled_data = unlabled_data.batch(UNLABELED_BATCH_SIZE).prefetch(AUTOTUNE)

# %%
labeled_train_data = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_labels))
labeled_train_data = labeled_train_data.map(preprocessing_jpg).prefetch(12)
labeled_train_data = labeled_train_data.shuffle(LABLED_DATA_SIZE_TRAIN)
labeled_train_data = labeled_train_data.batch(LABLED_BATCH_SIZE_TRAIN)

# %%
os.chdir("../test")

# %%
ls_Cr = glob.glob("Cr*.jpg")
ls_Sc = glob.glob("Sc*.jpg")
ls_In = glob.glob("In*.jpg")
ls_PS = glob.glob("PS*.jpg")
ls_RS = glob.glob("RS*.jpg")
ls_Pa = glob.glob("Pa*.jpg")

# %%
test_image_path, test_image_labels = ls_Cr + ls_Sc + ls_In + ls_PS + ls_RS + ls_Pa, [0]*len(ls_Cr) + [1]*len(ls_Sc) + [2]*len(ls_In) + [3]*len(ls_PS) + [4]*len(ls_RS) + [5]*len(ls_Pa)
test_image_path = [os.path.join("neu-metal-surface", "test", path) for path in test_image_path]

# %%
labeled_test_data = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_labels))
labeled_test_data = labeled_test_data.map(preprocessing_jpg)
#labeled_test_data = labeled_test_data.shuffle(LABLED_DATA_SIZE_TEST)
labeled_test_data = labeled_test_data.batch(LABLED_BATCH_SIZE_TEST)

# %%
zipped_data = tf.data.Dataset.zip((unlabled_data, labeled_train_data))

# %%
os.chdir("../val")

# %%
ls_Cr = glob.glob("Cr*.jpg")
ls_Sc = glob.glob("Sc*.jpg")
ls_In = glob.glob("In*.jpg")
ls_PS = glob.glob("PS*.jpg")
ls_RS = glob.glob("RS*.jpg")
ls_Pa = glob.glob("Pa*.jpg")

# %%
val_image_path, val_image_labels = ls_Cr + ls_Sc + ls_In + ls_PS + ls_RS + ls_Pa, [0]*len(ls_Cr) + [1]*len(ls_Sc) + [2]*len(ls_In) + [3]*len(ls_PS) + [4]*len(ls_RS) + [5]*len(ls_Pa)
val_image_path = [os.path.join("neu-metal-surface", "val", path) for path in val_image_path]

# %%
labeled_val_data = tf.data.Dataset.from_tensor_slices((val_image_path, val_image_labels))
labeled_val_data = labeled_val_data.map(preprocessing_jpg)
labeled_val_data = labeled_val_data.shuffle(LABLED_DATA_SIZE_VAL)
labeled_val_data = labeled_val_data.batch(LABLED_BATCH_SIZE_VAL)

# %%
os.chdir(os.path.join("..", ".."))



# %%
temperature = 0.1

contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {"min_area": 0.75, "brightness": 0.3, "jitter": 0.1}

# %%
image_channels = 3

# %%
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images



def get_augmenter(min_area, brightness, jitter):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], image_channels)),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            RandomColorAffine(brightness, jitter),
            
        ]
    )





# %%
width = 128
num_epochs = 100

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience= 15, restore_best_weights=True)

# %%


# %%
def channel_split(x, name=''):
    
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)


    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
       s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
       s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
       ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


# %%
def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x

# %%
def get_encoder(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3),
                 load_model=None,
                 num_shuffle_units=[1,1],
                 bottleneck_ratio=1):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],3)
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)


    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = width
    else:
        k = width
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(width, name='fc')(x)
        x = Activation('relu', name='RELU')(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model


# %%
baseline_model = keras.Sequential(
    [
        keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], image_channels)),
        get_augmenter(**classification_augmentation),
        get_encoder(),
        layers.Dense(6),
    ],
    name="baseline_model",
)
baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

baseline_history = baseline_model.fit(
    labeled_train_data, epochs=num_epochs, validation_data=labeled_val_data, callbacks = [early_stopping]
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(baseline_history.history["val_acc"]) * 100
    )
)



# %%
class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)
        self.encoder = get_encoder()

        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(width,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        )

        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(width,)), layers.Dense(16)], name="linear_probe"
        )

        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer


        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )


        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data


        images = tf.concat((unlabeled_images, labeled_images), axis=0)

        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)


        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )
        with tf.GradientTape() as tape:

            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data


        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)


        return {m.name: m.result() for m in self.metrics[2:]}



pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=tf.keras.optimizers.Adam(),
    probe_optimizer=tf.keras.optimizers.Adam(),
)

pretraining_history = pretraining_model.fit(
    zipped_data, epochs=num_epochs, validation_data=labeled_val_data
)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(pretraining_history.history["val_p_acc"]) * 100
    )
)

# %%
y_pred_base = baseline_model.predict(labeled_test_data, batch_size = None, verbose = 2,steps = None, callbacks = None)

# %%
y_pred_base = np.argmax(y_pred_base, axis = 1)

# %%
from sklearn.metrics import confusion_matrix

# %%
cm_baseline = confusion_matrix(test_image_labels, y_pred_base)

# %%
print(cm_baseline)

# %%
from sklearn.metrics import classification_report
report_base = classification_report(test_image_labels, y_pred_base)

# %%
print(report_base)

# %%


# %%
finetuning_model = keras.Sequential(
    [
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], image_channels)),
        get_augmenter(**classification_augmentation),
        pretraining_model.encoder,
        layers.Dense(6),
    ],
    name="finetuning_model",
)
finetuning_model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

finetuning_history = finetuning_model.fit(
    labeled_train_data, epochs=num_epochs, validation_data=labeled_val_data, callbacks=[early_stopping]
)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(finetuning_history.history["val_acc"]) * 100
    )
)

# %%
y_pred = finetuning_model.predict(labeled_test_data, batch_size = None, verbose = 2,steps = None, callbacks = None)

# %%
y_pred = np.argmax(y_pred, axis = 1)

# %%
cm = confusion_matrix(test_image_labels, y_pred)

# %%
print(cm)

# %%
from sklearn.metrics import classification_report
report = classification_report(test_image_labels, y_pred)

# %%
print(report)

# %%
#Curves
def plot_training_curves(pretraining_history, finetuning_history, baseline_history):
    for metric_key, metric_name in zip(["acc", "loss"], ["accuracy", "loss"]):
        plt.figure(figsize=(8, 5), dpi=100)
        plt.plot(
            baseline_history.history[f"val_{metric_key}"], label="supervised training"
        )
        plt.plot(
            pretraining_history.history[f"val_p_{metric_key}"],
            label="pretraining",
        )
        plt.plot(
            finetuning_history.history[f"val_{metric_key}"],
            label="finetuning",
        )
        plt.legend()
        plt.title(f"Classification {metric_name} during training")
        plt.xlabel("epochs")
        plt.ylabel(f"validation {metric_name}")


plot_training_curves(pretraining_history, finetuning_history, baseline_history)

# %%



