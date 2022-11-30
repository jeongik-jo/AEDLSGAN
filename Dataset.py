import tensorflow as tf
import HyperParameters as hp
import tensorflow_datasets as tfds


def load_dataset():
    dataset = tfds.load('celeb_a')
    if hp.train_data_size != -1:
        train_dataset = dataset['train'].shuffle(1000).take(hp.train_data_size)
    else:
        train_dataset = dataset['train'].shuffle(1000)

    if hp.test_data_size != -1:
        test_dataset = dataset['test'].take(hp.test_data_size)
    else:
        test_dataset = dataset['test']

    if hp.shuffle_test_dataset:
        test_dataset = test_dataset.shuffle(1000)

    train_dataset = train_dataset.batch(hp.batch_size, drop_remainder=True).map(_resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(hp.fid_batch_size, drop_remainder=True).map(_resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


@tf.function
def _resize_and_normalize(data):
    attributes = tf.cast(tf.stack([data['attributes'][attribute] for attribute in hp.attributes], axis=-1), dtype='float32')

    image = tf.image.resize(images=data['image'], size=[hp.image_resolution, hp.image_resolution])
    image = tf.cast(image, dtype='float32') / 127.5 - 1.0

    return {'attributes': attributes, 'image': image}
