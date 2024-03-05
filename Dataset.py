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

    train_dataset = train_dataset.batch(hp.batch_size, drop_remainder=True).map(_resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(hp.batch_size, drop_remainder=True).map(_resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


@tf.function
def _resize_and_normalize(data):
    ctg_vec = tf.cast(tf.stack([data['attributes'][att] for att in hp.atts], axis=-1), dtype='float32')

    img = tf.image.resize(images=data['image'], size=[hp.img_res, hp.img_res])
    img = tf.cast(img, dtype='float32') / 127.5 - 1.0

    return {'ctg_vec': ctg_vec, 'img': img}
