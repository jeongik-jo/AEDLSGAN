import tensorflow as tf
import HyperParameters as hp
import tensorflow_datasets as tfds


def load_celeb_dataset():
    dataset = tfds.load('celeb_a')
    if hp.train_data_size != -1:
        train_dataset = dataset['train'].shuffle(10000).take(hp.train_data_size)
    else:
        train_dataset = dataset['train'].shuffle(10000)

    if hp.test_data_size != -1:
        test_dataset = dataset['test'].shuffle(10000).take(hp.test_data_size)
    else:
        test_dataset = dataset['test'].shuffle(10000)

    train_dataset = train_dataset.batch(hp.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(hp.fid_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


@tf.function
def resize_and_normalize(images):
    images = tf.image.resize(images=images, size=[hp.image_resolution, hp.image_resolution])
    images = tf.cast(images, dtype='float32') / 127.5 - 1.0
    images = tf.image.random_flip_left_right(images)

    return images
