import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import tensorflow.keras as kr

generator_optimizer = kr.optimizers.Adam(learning_rate=0.003, beta_1=0.0, beta_2=0.99) # optimizer for decoder of generator
discriminator_optimizer = kr.optimizers.Adam(learning_rate=0.003, beta_1=0.0, beta_2=0.99) # optimizer for discriminator (and encoder)
lr_decay_rate = 0.9 # learning rate decay rate per epoch


image_resolution = 256

latent_vector_dim = 256

attributes = ['Bangs', 'Male', 'Smiling']
attribute_size = len(attributes)

enc_weight = 1.0
r1_weight = 10.0
var_vector_size = 512

batch_size = 8
save_image_size = 8
save_trans_value = 0.1

train_data_size = -1
test_data_size = -1
epochs = 25

load_model = False

evaluate_model = True
fid_batch_size = batch_size
epoch_per_evaluate = 1


def latent_dist_func(shape):
    return tf.random.normal(shape)
def latent_entropy_func(latent_scale_vector):
    return tf.reduce_sum(tf.math.log(latent_scale_vector * tf.sqrt(2.0 * 3.141592 * tf.exp(1.0))))
def latent_add_noises(latent_vectors, noise_size):
    noise = tf.random.normal([1, noise_size, latent_vector_dim], stddev=0.3)
    noised_vectors_sets = latent_vectors[:, tf.newaxis, :] + noise
    return noised_vectors_sets
