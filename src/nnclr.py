import random
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from simulated_data import create_simulated_data

tf.config.run_functions_eagerly(True)
import csv
import numpy as np
import os

# Augmentation

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def jitter(data, jitter_amount=0.01):
    return data + np.random.uniform(-jitter_amount, jitter_amount, data.shape)

def scale(data, scale_factor_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(*scale_factor_range)
    return data * scale_factor

def time_warp(data):
    time_steps = len(data)
    warp_amount = random.randint(-5, 5)
    if warp_amount > 0:
        return np.concatenate([data[warp_amount:], data[-warp_amount:]])
    else:
        return np.concatenate([data[:warp_amount], data[-warp_amount:]])

def window_slice(data, window_size=10):
    start = np.random.randint(0, max(len(data) - window_size, 1))
    return data[start:start + window_size]

def random_shift(data, shift_amount=5):
    shift = np.random.randint(-shift_amount, shift_amount)
    return np.roll(data, shift)

"""
## The NNCLR model for contrastive pre-training

We train an encoder on unlabeled images with a contrastive loss. A nonlinear projection
head is attached to the top of the encoder, as it improves the quality of representations
of the encoder.
"""

width = 128
num_epochs = 250

def time_series_encoder(time_series_feature_dimension):
    return keras.Sequential([
        layers.Input(shape=time_series_feature_dimension),  # Shape should match your time series length and channel
        layers.Conv1D(width, kernel_size=3, strides=2, activation="relu"),
        layers.Conv1D(width, kernel_size=3, strides=2, activation="relu"),
        layers.Flatten(),
        layers.Dense(width, activation="relu"),
    ])

def projection_head(time_series_feature_dimension):
    return keras.Sequential(
        [
            layers.Input(shape=(width,)),  # Input shape based on encoder output
            layers.Dense(width, activation='relu'),                # First dense layer
            layers.Dense(width),                                   # Second dense layer
        ],
        name='projection_head'
    )


class NNCLR(keras.Model):
    def __init__(self, temperature, queue_size, time_series_feature_dimension, time_column):
        super(NNCLR, self).__init__()
        self.time_column = time_column
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.encoder = time_series_encoder(time_series_feature_dimension)
        self.projection_head = projection_head(time_series_feature_dimension)

        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(width,)), layers.Dense(10)], name="linear_probe"
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

        batch_size = tf.cast(tf.shape(features_1, out_type=tf.int32)[0], tf.float32)
        cross_correlation = (
                tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
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

        if isinstance(data, tuple):
            (unlabeled_images, _), (labeled_images, labels) = data
            images = tf.concat((unlabeled_images, labeled_images), axis=0)
        else:
            images = data

        # Apply time series augmentations
        augmented_images_1 = self.apply_time_series_augmentation(images)
        augmented_images_2 = self.apply_time_series_augmentation(images)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)

            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)

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

        self.update_contrastive_accuracy(features_1, features_2)

        self.update_correlation_accuracy(features_1, features_2)

        if isinstance(data, tuple):
            preprocessed_images = self.classification_augmenter(labeled_images)

            with tf.GradientTape() as tape:
                features = self.encoder(preprocessed_images)
                class_logits = self.linear_probe(features)
                probe_loss = self.probe_loss(labels, class_logits)
            gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
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
        else:
            return {
                "c_loss": contrastive_loss,
                "c_acc": self.contrastive_accuracy.result(),
                "r_acc": self.correlation_accuracy.result(),
            }

    def apply_time_series_augmentation(self, time_series_data):
        # You can implement the logic to apply your defined augmentations to each time series
        augmented_data = []
        for data in time_series_data:
            augmented_sample = data
            augmented_sample = add_noise(augmented_sample)
            augmented_sample = jitter(augmented_sample)
            augmented_sample = scale(augmented_sample)
            augmented_sample = time_warp(augmented_sample)
            augmented_sample = window_slice(augmented_sample)
            augmented_sample = random_shift(augmented_sample)
            augmented_sample = resample_time_series(augmented_sample, self.time_column, len(data))
            augmented_data.append(augmented_sample)

        return np.array(augmented_data)


    # TODO Update for time series
    def test_step(self, data):
        images, labels = data

        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        features_1 = self.encoder(augmented_images_1)
        features_2 = self.encoder(augmented_images_2)
        projections_1 = self.projection_head(features_1)
        projections_2 = self.projection_head(features_2)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
        }


"""
## Pre-train NNCLR

We train the network using a `temperature` of 0.1 as suggested in the paper and
a `queue_size` of 10,000 as explained earlier. We use Adam as our contrastive and probe
optimizer. For this example we train the model for only 30 epochs but it should be
trained for more epochs for better performance.

The following two metrics can be used for monitoring the pretraining performance
which we also log (taken from
[this Keras example](https://keras.io/examples/vision/semisupervised_simclr/#selfsupervised-model-for-contrastive-pretraining)):

- Contrastive accuracy: self-supervised metric, the ratio of cases in which the
representation of an image is more similar to its differently augmented version's one,
than to the representation of any other image in the current batch. Self-supervised
metrics can be used for hyperparameter tuning even in the case when there are no labeled
examples.
- Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised
classifiers. It is computed as the accuracy of a logistic regression classifier trained
on top of the encoder's features. In our case, this is done by training a single dense
layer on top of the frozen encoder. Note that contrary to traditional approach where the
classifier is trained after the pretraining phase, in this example we train it during
pretraining. This might slightly decrease its accuracy, but that way we can monitor its
value during training, which helps with experimentation and debugging.
"""

import pandas as pd

def resample_time_series(subset, time_column, target_length):
    """
    Resample the given time series subset to the target length using interpolation
    based on the values in the time_column.
    """
    # Get the original time values from the time_column
    original_time = subset.T[time_column]
    target_time = np.linspace(original_time.min(), original_time.max(), target_length)

    resampled_subset = pd.DataFrame(index=target_time)

    for column in range(subset.shape[1]):
        if column != time_column:  # Do not resample the time_column itself
            # Interpolate each column based on original time values and target time
            resampled_subset[column] = np.interp(target_time, original_time, subset.T[column])

    # Include the resampled time column as well
    resampled_subset[time_column] = target_time

    return resampled_subset



import numpy as np

def extract_time_series_subsets(time_series_df, time_column, annotations, target_length=50):
    """
    Extract subsets of the time series based on the 'index', time_column, x_min, and x_max annotations
    and resample them to a target length.
    """
    subsets = []
    for annotation in annotations:
        # Extract the specific time series based on annotation['index']
        df_subset = time_series_df.loc[annotation['index']]

        # Filter rows within the x_min and x_max range using the specified time_column
        subset = df_subset[
            (df_subset[time_column] >= annotation['x_min']) &
            (df_subset[time_column] <= annotation['x_max'])
        ]

        # Resample the subset to the target length if it's not empty
        if subset.dropna().values.shape[0] > 0:
            resampled_subset = resample_time_series(subset.values, subset.columns.get_loc(time_column), target_length)
            subsets.append(resampled_subset)

    return np.array(subsets)

import json
import os

# Load annotations and preprocess categorical columns
def load_annotations():
    annotations = []
    if os.path.exists('annotations.json'):
        with open('annotations.json', 'r') as jsonfile:
            data = json.load(jsonfile)
            for entry in data:
                annotation_data = {
                    'index': tuple(entry['index']),
                    'x_min': float(entry['x_min']),
                    'x_max': float(entry['x_max']),
                    'hurry': 1 if entry['hurry'] == 'Yes' else 0,
                    'frustrateion': 1 if entry['frustration'] == 'Yes' else 0,
                    'surprise': 1 if entry['surprise'] == 'Yes' else 0,
                    'risk_outcome': 1 if entry['risk_outcome'] == 'Accident' else 0,
                }
                annotations.append(annotation_data)
    return annotations


def prepare_time_series_dataset(time_series_subsets, batch_size):
    # Convert the list of subsets to a tensor dataset
    dataset = tf.data.Dataset.from_tensor_slices(time_series_subsets)

    dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(buffer_size=len(time_series_subsets))

    return dataset


def fit_nnclr_model(time_series_df, time_colmn, annotations, num_epochs=250, target_length=50, batch_size=16):
    time_series_subsets = extract_time_series_subsets(time_series_df, time_colmn, annotations, target_length)
    train_dataset = prepare_time_series_dataset(time_series_subsets, batch_size)
    # Define your model (as provided in your original code)
    model = NNCLR(temperature=0.25, queue_size=100, time_series_feature_dimension=time_series_subsets.shape[1:], time_column=time_series_df.columns.get_loc(time_colmn))
    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
    )
    # Train the model
    pretrain_history = model.fit(
        train_dataset, epochs=num_epochs,  # validation_data=train_dataset if needed
    )
    return model

def prepare_model_to_load(time_series_df, time_colmn, annotations, target_length=50,):
    time_series_subsets = extract_time_series_subsets(time_series_df, time_colmn, annotations, target_length)

    model = NNCLR(temperature=0.25, queue_size=100,time_series_feature_dimension=time_series_subsets.shape[1:], time_column=time_series_df.columns.get_loc(time_colmn))

    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
    )
    model.built = True

    return model

def get_similarities(model, time_series_df, time_colmn, annotations, target_length=50, batch_size=16):
    time_series_subsets = extract_time_series_subsets(time_series_df, time_colmn, annotations, target_length)
    predict_dataset = prepare_time_series_dataset(time_series_subsets, batch_size)

    feature_vectors = model.encoder.predict(predict_dataset, batch_size = 16, verbose = 1)
    # Normalize the feature vectores.
    feature_vectors = tf.math.l2_normalize(feature_vectors, -1)

    similarities = tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
    similarities = similarities.numpy()
    return similarities

if __name__ == '__main__':
    import random

    # Generate some sample time series data as a pandas DataFrame
    time_series_df, time_column = create_simulated_data()

    # Load annotations from the json
    annotations = load_annotations()

    # Extract and resample the time series subsets based on the annotations
    target_length = 50  # All subsets will be resampled to this length
    batch_size = 16  # Adjust based on your requirements

    model = fit_nnclr_model(time_series_df, time_column, annotations, target_length=target_length, batch_size=batch_size)
