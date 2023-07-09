# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers
import math
import keras


# %%
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
import matplotlib.pyplot as plt


# %%
os.chdir("pretraining_new/")


# %%
!rm * .jpg~


# %%
unlabled_img_names = os.listdir()
unlabled_img_labels = [-1]*len(unlabled_img_names)
unlabled_img_names = [os.path.join("pretraining_new", path)
                      for path in unlabled_img_names]


# %%
BATCH_SIZE = 32
#IMAGE_SIZE = (128, 800)
IMAGE_SIZE = (200, 500)
UNLABLED_DATA_SIZE = 21222
LABLED_DATA_SIZE_TRAIN = 1260
LABLED_DATA_SIZE_TEST = 270
LABLED_DATA_SIZE_VAL = 270
queue_size = 10000
STEPS_PER_EPOCH = (UNLABLED_DATA_SIZE + LABLED_DATA_SIZE_TRAIN) // BATCH_SIZE
UNLABELED_BATCH_SIZE = UNLABLED_DATA_SIZE // STEPS_PER_EPOCH
LABLED_BATCH_SIZE_TRAIN = LABLED_DATA_SIZE_TRAIN // STEPS_PER_EPOCH
LABLED_BATCH_SIZE_TEST = LABLED_DATA_SIZE_TEST//(STEPS_PER_EPOCH//10)
LABLED_BATCH_SIZE_VAL = LABLED_DATA_SIZE_VAL//(STEPS_PER_EPOCH//10)


print(STEPS_PER_EPOCH, UNLABELED_BATCH_SIZE, LABLED_BATCH_SIZE_TRAIN,
      LABLED_BATCH_SIZE_TEST, LABLED_BATCH_SIZE_VAL)

AUTOTUNE = tf.data.AUTOTUNE


# %%
def decode_jpg(img_name):
    img = tf.io.read_file(img_name)
    img = tf.image.decode_jpeg(img, channels=3)
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
train_image_path, train_image_labels = ls_Cr + ls_Sc + ls_In + ls_PS + ls_RS + \
    ls_Pa, [0]*len(ls_Cr) + [1]*len(ls_Sc) + [2]*len(ls_In) + \
    [3]*len(ls_PS) + [4]*len(ls_RS) + [5]*len(ls_Pa)


# %%
train_image_path = [os.path.join(
    "neu-metal-surface", "train", path) for path in train_image_path]
len(train_image_labels) == len(train_image_path)


# %%
def decode_bmp(img_name):
    img = tf.io.read_file(img_name)
    img = tf.image.decode_bmp(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def preprocessing_bmp(img_name, labels):
    return decode_bmp(img_name), labels


# %%
unlabled_data = tf.data.Dataset.from_tensor_slices(
    (unlabled_img_names, unlabled_img_labels))
unlabled_data = unlabled_data.map(preprocessing_jpg)
unlabled_data = unlabled_data.shuffle(UNLABLED_DATA_SIZE)
unlabled_data = unlabled_data.batch(
    UNLABELED_BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)


# %%
labeled_train_data = tf.data.Dataset.from_tensor_slices(
    (train_image_path, train_image_labels))
labeled_train_data = labeled_train_data.map(preprocessing_jpg).prefetch(12)
labeled_train_data = labeled_train_data.shuffle(LABLED_DATA_SIZE_TRAIN)
labeled_train_data = labeled_train_data.batch(
    LABLED_BATCH_SIZE_TRAIN, drop_remainder=True)


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
test_image_path, test_image_labels = ls_Cr + ls_Sc + ls_In + ls_PS + ls_RS + \
    ls_Pa, [0]*len(ls_Cr) + [1]*len(ls_Sc) + [2]*len(ls_In) + \
    [3]*len(ls_PS) + [4]*len(ls_RS) + [5]*len(ls_Pa)
test_image_path = [os.path.join("neu-metal-surface", "test", path)
                   for path in test_image_path]


# %%
labeled_test_data = tf.data.Dataset.from_tensor_slices(
    (test_image_path, test_image_labels))
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
val_image_path, val_image_labels = ls_Cr + ls_Sc + ls_In + ls_PS + ls_RS + \
    ls_Pa, [0]*len(ls_Cr) + [1]*len(ls_Sc) + [2]*len(ls_In) + \
    [3]*len(ls_PS) + [4]*len(ls_RS) + [5]*len(ls_Pa)
val_image_path = [os.path.join("neu-metal-surface", "val", path)
                  for path in val_image_path]


# %%
labeled_val_data = tf.data.Dataset.from_tensor_slices(
    (val_image_path, val_image_labels))
labeled_val_data = labeled_val_data.map(preprocessing_jpg)
labeled_val_data = labeled_val_data.shuffle(LABLED_DATA_SIZE_VAL)
labeled_val_data = labeled_val_data.batch(LABLED_BATCH_SIZE_VAL)


# %%
os.chdir(os.path.join("..", ".."))


# %%
temperature = 0.1
queue_size = 10000
contrastive_augmenter = {
    "brightness": 0.5,
    "name": "contrastive_augmenter",
    "scale": (0.2, 1.0),
}
classification_augmenter = {
    "brightness": 0.2,
    "name": "classification_augmenter",
    "scale": (0.5, 1.0),
}


# %%
class RandomResizedCrop(layers.Layer):
    def __init__(self, scale, ratio):
        super(RandomResizedCrop, self).__init__()
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        random_scales = tf.random.uniform(
            (batch_size,), self.scale[0], self.scale[1])
        random_ratios = tf.exp(
            tf.random.uniform(
                (batch_size,), self.log_ratio[0], self.log_ratio[1])
        )

        new_heights = tf.clip_by_value(
            tf.sqrt(random_scales / random_ratios), 0, 1)
        new_widths = tf.clip_by_value(
            tf.sqrt(random_scales * random_ratios), 0, 1)
        height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
        width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            images, bounding_boxes, tf.range(batch_size), (height, width)
        )
        return images


# %%
class RandomBrightness(layers.Layer):
    def __init__(self, brightness):
        super(RandomBrightness, self).__init__()
        self.brightness = brightness

    def blend(self, images_1, images_2, ratios):
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):

        return self.blend(
            images,
            0,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 -
                self.brightness, 1 + self.brightness
            ),
        )

    def call(self, images):
        images = self.random_brightness(images)
        return images


# %%
def get_augmenter(brightness, name, scale):
    return keras.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], image_channels)),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=scale, ratio=(3 / 4, 4 / 3)),
            RandomBrightness(brightness=brightness),
        ],
        name=name,
    )


# %%
width = 128
image_channels = 3
num_epochs = 100


# %%
def get_encoder():
    return keras.Sequential(
        [
            keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], image_channels)),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    )


# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=15, restore_best_weights=True)


# %% [markdown]
# Baseline
# 

# %%
baseline_model = keras.Sequential(
    [
        keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], image_channels)),
        get_augmenter(**classification_augmenter),
        get_encoder(),
        layers.Dense(6),
    ],
    name="baseline_model",
)
baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

baseline_history = baseline_model.fit(
    labeled_train_data, epochs=num_epochs, validation_data=labeled_val_data, callbacks=[
        early_stopping]
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(baseline_history.history["val_acc"]) * 100
    )
)


# %%
y_pred_base = baseline_model.predict(
    labeled_test_data, batch_size=None, verbose=2, steps=None, callbacks=None)


# %%
y_pred_base = np.argmax(y_pred_base, axis=1)


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


# %%
class NNCLR(keras.Model):
    def __init__(
        self, temperature, queue_size,
    ):
        super(NNCLR, self).__init__()
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        self.contrastive_augmenter = get_augmenter(**contrastive_augmenter)
        self.classification_augmenter = get_augmenter(
            **classification_augmenter)
        self.encoder = get_encoder()
        self.projection_head = keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        )
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(width,)), layers.Dense(16)], name="linear_probe"
        )
        self.temperature = temperature

        feature_dimensions = self.encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super(NNCLR, self).compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1, out_type=tf.int32)[0]
        # added
        batch_size = tf.cast(batch_size, dtype=tf.float32, name=None)
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(
                cross_correlation)], axis=0),
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2)
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
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)
        preprocessed_images = self.classification_augmenter(labeled_images)

        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(
            probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }

    def test_step(self, data):
        labeled_images, labels = data

        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}


# %%
model = NNCLR(temperature=temperature, queue_size=queue_size)
model.compile(
    contrastive_optimizer=tf.keras.optimizers.Adam(),
    probe_optimizer=tf.keras.optimizers.Adam(),
)
pretrain_history = model.fit(
    zipped_data, epochs=num_epochs, validation_data=labeled_val_data
)


# %%
finetuning_model = keras.Sequential(
    [
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], image_channels)),
        get_augmenter(**classification_augmenter),
        model.encoder,
        layers.Dense(6),
    ],
    name="finetuning_model",
)
finetuning_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

finetuning_history = finetuning_model.fit(
    labeled_train_data, epochs=num_epochs, validation_data=labeled_val_data, callbacks=[
        early_stopping]
)


# %%
y_pred = finetuning_model.predict(
    labeled_test_data, batch_size=None, verbose=2, steps=None, callbacks=None)


# %%
y_pred = np.argmax(y_pred, axis=1)


# %%
from sklearn.metrics import confusion_matrix


# %%
cm = confusion_matrix(test_image_labels, y_pred)


# %%
print(cm)


# %%
from sklearn.metrics import classification_report
report = classification_report(test_image_labels, y_pred)


# %%
print(report)


# %% [markdown]
# 

# %%
def plot_training_curves(pretraining_history, finetuning_history, baseline_history):
    for metric_key, metric_name in zip(["acc", "loss"], ["accuracy", "loss"]):
        plt.figure(figsize=(8, 5), dpi=100)
        plt.plot(
            baseline_history.history[f"val_{metric_key}"], label="Baseline"
        )
        plt.plot(
            pretraining_history.history[f"val_p_{metric_key}"],
            label="Pretraining",
        )
        plt.plot(
            finetuning_history.history[f"val_{metric_key}"],
            label="Downstream/Finetuning",
        )
        plt.legend()
        plt.title(f"Classification {metric_name}")
        plt.xlabel("epochs")
        plt.ylabel(f"validation {metric_name}")


plot_training_curves(pretrain_history, finetuning_history, baseline_history)


# %%



