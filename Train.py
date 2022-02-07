import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import Dataset


@tf.function
def _train_step(generator: kr.Model, discriminator: kr.Model, data: tf.Tensor, var_vectors, svm_weights):
    with tf.GradientTape(persistent=True) as tape:
        real_images = Dataset.resize_and_normalize(data['image'])
        batch_size = real_images.shape[0]

        real_condition_vectors = tf.cast(tf.stack([data['attributes'][attribute] for attribute in hp.attributes], axis=-1), dtype='float32')
        real_condition_vectors = tf.concat([real_condition_vectors, 1 - real_condition_vectors], axis=-1)

        latent_scale_vector = tf.sqrt(tf.reduce_mean(var_vectors, axis=0, keepdims=True))
        latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32')) * latent_scale_vector / tf.norm(latent_scale_vector, axis=-1, keepdims=True)
        latent_vectors = hp.latent_dist_func([batch_size, hp.latent_vector_dim])

        fake_condition_vectors = tf.matmul(latent_vectors * latent_scale_vector, svm_weights)
        fake_condition_vectors = tf.where(fake_condition_vectors < 0, 0.0, 1.0)
        fake_condition_vectors = tf.concat([fake_condition_vectors, 1 - fake_condition_vectors], axis=-1)

        fake_images = generator(latent_vectors * latent_scale_vector, training=True)
        fake_adv_vectors, rec_latent_vectors = discriminator(fake_images, training=True)
        enc_losses = tf.reduce_mean(tf.square(rec_latent_vectors - latent_vectors) * tf.square(latent_scale_vector), axis=-1)

        with tf.GradientTape() as inner_tape:
            inner_tape.watch(real_images)
            real_adv_vectors, _ = discriminator(real_images, training=True)
            real_adv_score = tf.reduce_mean(real_adv_vectors * real_condition_vectors, axis=-1) * 2
        real_gradients = inner_tape.gradient(real_adv_score, real_images)
        r1_regs = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])

        discriminator_adv_losses = tf.reduce_mean(tf.nn.softplus(-real_adv_vectors) * real_condition_vectors
                                                  + tf.nn.softplus(fake_adv_vectors) * fake_condition_vectors, axis=-1) * 2
        discriminator_losses = discriminator_adv_losses + hp.enc_weight * enc_losses + hp.r1_weight * r1_regs
        discriminator_loss = tf.reduce_mean(discriminator_losses)
        generator_adv_losses = tf.reduce_mean(tf.nn.softplus(-fake_adv_vectors) * fake_condition_vectors, axis=-1) * 2
        generator_losses = generator_adv_losses + hp.enc_weight * enc_losses
        generator_loss = tf.reduce_mean(generator_losses)

    var_vectors = tf.concat([var_vectors[1:], tf.reduce_mean(tf.square(rec_latent_vectors), axis=0, keepdims=True)], axis=0)
    hp.generator_optimizer.apply_gradients(
        zip(tape.gradient(generator_loss, generator.trainable_variables),
            generator.trainable_variables)
    )

    hp.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(discriminator_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    del tape

    return tf.reduce_mean(enc_losses), var_vectors


def train(generator: kr.Model, discriminator: kr.Model, dataset, var_vectors, svm_weights):
    enc_losses = []
    for data in dataset:
        enc_loss, var_vectors = _train_step(generator, discriminator, data, var_vectors, svm_weights)
        enc_losses.append(enc_loss)
    mean_enc_loss = tf.reduce_mean(enc_losses)

    return mean_enc_loss, var_vectors
