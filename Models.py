import tensorflow as tf
import tensorflow.keras as kr
import Dataset
import Layers
import os
import HyperParameters as hp
import numpy as np


class Generator(object):
    def build_model(self):
        latent_vector = kr.Input([hp.latent_vector_dim])
        fake_image = Layers.Generator()(latent_vector)
        return kr.Model(latent_vector, fake_image)

    def __init__(self):
        self.model = self.build_model()
        self.var_vectors = tf.ones([hp.var_vector_size, hp.latent_vector_dim])
        self.svm_weights = tf.random.normal([hp.latent_vector_dim, hp.attribute_size])

    def save_images(self, test_dataset: tf.data.Dataset, discriminator: kr.Model, latent_scale_vector, epoch):
        def save_fake_images():
            if not os.path.exists('./results/fake_images'):
                os.makedirs('./results/fake_images')
            save_images = []
            save_labels = []
            for _ in range(hp.save_image_size):
                latent_vectors = hp.latent_dist_func([hp.save_image_size, hp.latent_vector_dim])
                condition_vectors = tf.matmul(latent_vectors * latent_scale_vector, self.svm_weights)
                condition_vectors = tf.where(condition_vectors < 0.0, 0.0, 1.0)
                save_labels.append(tf.reduce_sum(2.0 ** tf.cast(tf.range(hp.attribute_size), dtype='float32') * condition_vectors, axis=-1))

                fake_images = self.model(latent_vectors * latent_scale_vector, training=False)
                save_images.append(np.hstack(fake_images))

            kr.preprocessing.image.save_img(path='./results/fake_images/fake_%d.png' % epoch,
                                            x=tf.clip_by_value(np.vstack(save_images), clip_value_min=-1, clip_value_max=1))
            np.savetxt('./results/fake_images/fake_%d_label.txt' % epoch, np.array(save_labels), fmt='%d')
        save_fake_images()
        # --------------------------------------------------------------------------------------------------------------
        def save_rec_images(is_real):
            if is_real:
                path = './results/real_rec_images'
            else:
                path = './results/fake_rec_images'
            if not os.path.exists(path):
                os.makedirs(path)

            save_images = []
            for data in test_dataset.take(hp.save_image_size // 2):
                if is_real:
                    input_images = Dataset.resize_and_normalize(data['image'][:hp.save_image_size])
                else:
                    input_images = self.model(hp.latent_dist_func([hp.save_image_size, hp.latent_vector_dim]) * latent_scale_vector, training=False)

                _, rec_latent_vectors = discriminator(input_images, training=False)
                rec_images = self.model(rec_latent_vectors * latent_scale_vector, training=False)

                save_images.append(np.vstack(input_images))
                save_images.append(np.vstack(rec_images))
                save_images.append(tf.ones([np.vstack(input_images).shape[0], 5, 3]))

            save_image = tf.clip_by_value(np.hstack(save_images), clip_value_min=-1, clip_value_max=1)
            if is_real:
                kr.preprocessing.image.save_img(path=path + '/real_rec_%d.png' % epoch, x=save_image)
            else:
                kr.preprocessing.image.save_img(path=path + '/fake_rec_%d.png' % epoch, x=save_image)

        save_rec_images(True)
        save_rec_images(False)

        # --------------------------------------------------------------------------------------------------------------
        def save_latent_noised_images():
            if not os.path.exists('./results/latent_noised_images'):
                os.makedirs('./results/latent_noised_images')
            images = []
            for data in test_dataset.take(1):
                real_images = Dataset.resize_and_normalize(data['image'][:hp.save_image_size])
                _, rec_latent_vectors = discriminator(real_images, training=False)
                reconstructed_images = self.model(rec_latent_vectors * latent_scale_vector, training=False)
                noised_vectors_sets = hp.latent_add_noises(rec_latent_vectors, hp.save_image_size)

                for i in range(hp.save_image_size):
                    images.append(np.hstack([
                        real_images[i],
                        reconstructed_images[i],
                        tf.ones([hp.image_resolution, 5, 3]),
                        np.hstack(self.model(noised_vectors_sets[i] * latent_scale_vector, training=False))]))

            kr.preprocessing.image.save_img(path='./results/latent_noised_images/latent_noised_%d.png' % epoch,
                                            x=tf.clip_by_value(np.vstack(images), clip_value_min=-1, clip_value_max=1))
        save_latent_noised_images()
        # --------------------------------------------------------------------------------------------------------------
        def save_rec_trans_images(is_real):
            if is_real:
                path = './results/real_rec_trans_images'
            else:
                path = './results/fake_rec_trans_images'
            if not os.path.exists(path):
                os.makedirs(path)

            if is_real:
                for data in test_dataset.take(1):
                    input_images = Dataset.resize_and_normalize(data['image'][:hp.save_image_size])
            else:
                latent_vectors = hp.latent_dist_func([hp.save_image_size, hp.latent_vector_dim])
                input_images = self.model(latent_vectors * latent_scale_vector, training=False)

            for i in range(hp.attribute_size):
                save_images = []
                for j in range(hp.save_image_size):
                    _, rec_latent_vector = discriminator(input_images[j][tf.newaxis], training=False)
                    rec_image = self.model(rec_latent_vector * latent_scale_vector, training=False)

                    weight = self.svm_weights[tf.newaxis, :, i]
                    condition_score = tf.reduce_mean(rec_latent_vector * latent_scale_vector * weight, keepdims=True)
                    k = hp.latent_vector_dim * (condition_score - tf.linspace(-hp.save_trans_value, hp.save_trans_value, hp.save_image_size)[:, tf.newaxis]) / tf.reduce_sum(tf.square(weight), keepdims=True)
                    trans_images = self.model(rec_latent_vector * latent_scale_vector - k * weight, training=False)
                    save_images.append(np.hstack([
                        input_images[j],
                        rec_image[0],
                        tf.ones([hp.image_resolution, 5, 3]),
                        np.hstack(trans_images),
                    ]))
                save_image = tf.clip_by_value(np.vstack(save_images), clip_value_min=-1, clip_value_max=1)
                if is_real:
                    kr.preprocessing.image.save_img(path=path + '/real_%d_%d.png' % (epoch, i), x=save_image)
                else:
                    kr.preprocessing.image.save_img(path=path + '/fake_%d_%d.png' % (epoch, i), x=save_image)

        save_rec_trans_images(True)
        save_rec_trans_images(False)
        # --------------------------------------------------------------------------------------------------------------
    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        self.model.save_weights('./models/generator.h5')
        np.save('./models/var_vectors.npy', self.var_vectors)
        np.save('./models/svm_weights.npy', self.svm_weights)

    def load(self):
        self.model.load_weights('./models/generator.h5')
        self.var_vectors = np.load('./models/var_vectors.npy')
        self.svm_weights = np.load('./models/svm_weights.npy')


class Discriminator(object):
    def build_model(self):
        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 3])
        adv_vector, latent_vector = Layers.Discriminator()(input_image)
        return kr.Model(input_image, [adv_vector, latent_vector])

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        self.model.save_weights('./models/discriminator.h5')

    def load(self):
        self.model.load_weights('./models/discriminator.h5')

