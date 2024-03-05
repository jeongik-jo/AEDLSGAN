import tensorflow as tf
from tensorflow import keras as kr
import Layers
import os
import HyperParameters as hp
import numpy as np


class Generator(object):
    def build_model(self):
        ltn_vec = kr.Input([hp.ltn_dim])
        return kr.Model(ltn_vec, Layers.Generator()(ltn_vec))

    def __init__(self):
        self.model = self.build_model()
        self.save_ltn_vecs = [hp.ltn_dist_func(hp.batch_size) for _ in range(hp.batch_size)]

    def save_imgs(self, dis: kr.Model, dataset: tf.data.Dataset, epoch):
        ltn_scl_vecs = hp.get_ltn_scl_vecs()
        if not os.path.exists('results/samples'):
            os.makedirs('results/samples')
        # --------------------------------------------------------------------------------------------------------------
        def save_fake_imgs():
            path = 'results/samples/fake_images'
            if not os.path.exists(path):
                os.makedirs(path)

            imgs = []
            for i in range(hp.batch_size):
                imgs.append(np.hstack(self.model(self.save_ltn_vecs[i] * ltn_scl_vecs)))

            kr.preprocessing.image.save_img(path=path + '/fake_%d.png' % epoch,
                                            x=tf.clip_by_value(np.vstack(imgs), clip_value_min=-1, clip_value_max=1))
        save_fake_imgs()
        # --------------------------------------------------------------------------------------------------------------
        def save_rec_imgs(is_real):
            if is_real:
                path = 'results/samples/real_rec_images'
            else:
                path = 'results/samples/fake_rec_images'
            if not os.path.exists(path):
                os.makedirs(path)

            imgs = []
            for real_data in dataset.take(hp.batch_size // 2):
                if is_real:
                    inp_imgs = real_data['img'][:hp.batch_size]
                else:
                    inp_imgs = self.model(hp.ltn_dist_func(hp.batch_size) * ltn_scl_vecs)
                rec_imgs = self.model(dis(inp_imgs)[1] * ltn_scl_vecs)

                imgs.append(np.vstack(inp_imgs))
                imgs.append(np.vstack(rec_imgs))
                imgs.append(tf.ones([np.vstack(inp_imgs).shape[0], 5, hp.img_chn]))

            imgs = tf.clip_by_value(np.hstack(imgs), clip_value_min=-1, clip_value_max=1)
            if is_real:
                kr.preprocessing.image.save_img(path=path + '/real_rec_%d.png' % epoch, x=imgs)
            else:
                kr.preprocessing.image.save_img(path=path + '/fake_rec_%d.png' % epoch, x=imgs)

        save_rec_imgs(True)
        save_rec_imgs(False)
        # --------------------------------------------------------------------------------------------------------------
        def save_int_imgs():
            path = 'results/samples/latent_interpolation'
            if not os.path.exists(path):
                os.makedirs(path)

            indexes = tf.argsort(ltn_scl_vecs[0], axis=-1, direction='DESCENDING')
            int_vals = tf.linspace(-hp.ltn_int_val, hp.ltn_int_val, hp.batch_size)[:, tf.newaxis]
            ltn_vecs = hp.ltn_dist_func(hp.batch_size)
            for i in range(hp.batch_size):
                imgs = []
                mask = tf.one_hot(indexes[i], axis=-1, depth=hp.ltn_dim)[tf.newaxis]
                for j in range(hp.batch_size):
                    int_ltn_vecs = ltn_vecs[j][tf.newaxis] * (1 - mask) + int_vals * mask
                    imgs.append(np.hstack(self.model(int_ltn_vecs * ltn_scl_vecs)))

                kr.preprocessing.image.save_img(
                    path=path + '/latent_interpolation_%d_%d.png' % (epoch, i),
                    x=tf.clip_by_value(np.vstack(imgs), clip_value_min=-1, clip_value_max=1))

        save_int_imgs()
        # --------------------------------------------------------------------------------------------------------------
        def save_trs_imgs(is_real):
            if is_real:
                path = 'results/samples/real_attribute_edit_images'
            else:
                path = 'results/samples/fake_attribute_edit_images'
            if not os.path.exists(path):
                os.makedirs(path)

            for real_data in dataset.take(1):
                if is_real:
                    inp_imgs = real_data['img'][:hp.batch_size]
                else:
                    inp_imgs = self.model(hp.ltn_dist_func(hp.batch_size) * ltn_scl_vecs)

            for i in range(hp.ctg_dim):
                imgs = []
                for j in range(hp.batch_size):
                    _, rec_ltn_vecs = dis(tf.tile(inp_imgs[j][tf.newaxis], [hp.batch_size, 1, 1, 1]))

                    rec_image = self.model(rec_ltn_vecs * ltn_scl_vecs)[0]
                    rec_ltn_vecs = rec_ltn_vecs[0][tf.newaxis]

                    w = hp.cla_w[:, i]
                    k = tf.reduce_sum(rec_ltn_vecs * ltn_scl_vecs * w)
                    k = k - tf.cast(tf.linspace(-hp.trs_val, hp.trs_val, hp.batch_size), 'float32')[:, tf.newaxis] * tf.sqrt(tf.cast(hp.ltn_dim, 'float32'))
                    k = k * w / tf.reduce_sum(tf.square(w))

                    trs_imgs = self.model(rec_ltn_vecs * ltn_scl_vecs - k)

                    imgs.append(np.hstack([
                        inp_imgs[j],
                        rec_image,
                        tf.ones([hp.img_res, 5, 3]),
                        np.hstack(trs_imgs),
                    ]))

                imgs = tf.clip_by_value(np.vstack(imgs), clip_value_min=-1, clip_value_max=1)
                if is_real:
                    kr.preprocessing.image.save_img(path=path + '/real_trans_%d_%d.png' % (epoch, i), x=imgs)
                else:
                    kr.preprocessing.image.save_img(path=path + '/fake_trans_%d_%d.png' % (epoch, i), x=imgs)

        save_trs_imgs(True)
        save_trs_imgs(False)
        # --------------------------------------------------------------------------------------------------------------
    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/generator.h5')
        np.save('models/latent_var_trace.npy', hp.ltn_var_trace)
        np.save('models/cla_w.npy', hp.cla_w)

    def load(self):
        self.model.load_weights('models/generator.h5')
        hp.ltn_var_trace.assign(np.load('models/latent_var_trace.npy'))
        hp.cla_w.assign(np.load('models/cla_w.npy'))

    def to_ema(self):
        self.train_w = [tf.constant(w) for w in self.model.trainable_variables]
        hp.gen_opt.finalize_variable_values(self.model.trainable_variables)


    def to_train(self):
        for ema_w, train_w in zip(self.model.trainable_variables, self.train_w):
            ema_w.assign(train_w)


class Discriminator(object):
    def build_model(self):
        inp_img = kr.Input([hp.img_res, hp.img_res, hp.img_chn])
        return kr.Model(inp_img, Layers.Discriminator()(inp_img))

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/discriminator.h5')

    def load(self):
        self.model.load_weights('models/discriminator.h5')

    def to_ema(self):
        self.train_w = [tf.constant(w) for w in self.model.trainable_variables]
        hp.dis_opt.finalize_variable_values(self.model.trainable_variables)

    def to_train(self):
        for ema_w, train_w in zip(self.model.trainable_variables, self.train_w):
            ema_w.assign(train_w)