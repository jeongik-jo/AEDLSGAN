import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _train_step(dis: kr.Model, gen: kr.Model, real_data):
    ltn_scl_vecs = hp.get_ltn_scl_vecs()
    real_imgs = real_data['img']
    real_ctg_vecs = real_data['ctg_vec']
    real_ctg_vecs = tf.concat([real_ctg_vecs, 1 - real_ctg_vecs], axis=-1)
    batch_size = real_imgs.shape[0]

    ltn_vecs = hp.ltn_dist_func(batch_size)
    ctg_vecs = (ltn_vecs * ltn_scl_vecs) @ hp.cla_w
    ctg_vecs = tf.where(ctg_vecs > 0.0, 1.0, 0.0)
    ctg_vecs = tf.concat([ctg_vecs, 1 - ctg_vecs], axis=-1)

    fake_imgs = gen(ltn_vecs * ltn_scl_vecs)

    with tf.GradientTape() as dis_tape:
        with tf.GradientTape() as reg_tape:
            reg_tape.watch(real_imgs)
            real_adv_vecs, _ = dis(real_imgs)
            real_scores = tf.reduce_sum(real_adv_vecs * real_ctg_vecs, axis=-1) / hp.ctg_dim
        reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(real_scores, real_imgs)), axis=[1, 2, 3]))
        fake_adv_vecs, rec_ltn_vecs = dis(fake_imgs)

        enc_loss = tf.reduce_mean(tf.square((ltn_vecs - rec_ltn_vecs) * ltn_scl_vecs))

        dis_adv_losses = tf.reduce_sum(tf.nn.softplus(-real_adv_vecs) * real_ctg_vecs + tf.nn.softplus(fake_adv_vecs) * ctg_vecs, axis=-1) / hp.ctg_dim
        dis_adv_loss = tf.reduce_mean(dis_adv_losses)
        dis_loss = dis_adv_loss + hp.enc_w * enc_loss + hp.reg_w * reg_loss

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_tape)
    rec_ltn_traces = rec_ltn_vecs

    ltn_vecs = hp.ltn_dist_func(batch_size)
    ctg_vecs = (ltn_vecs * ltn_scl_vecs) @ hp.cla_w
    ctg_vecs = tf.where(ctg_vecs > 0.0, 1.0, 0.0)
    ctg_vecs = tf.concat([ctg_vecs, 1 - ctg_vecs], axis=-1)

    with tf.GradientTape() as gen_tape:
        fake_imgs = gen(ltn_vecs * ltn_scl_vecs)
        fake_adv_vecs, rec_ltn_vecs = dis(fake_imgs)

        enc_loss = tf.reduce_mean(tf.square((ltn_vecs - rec_ltn_vecs) * ltn_scl_vecs))

        gen_adv_losses = tf.reduce_sum(tf.nn.softplus(-fake_adv_vecs) * ctg_vecs, axis=-1) / hp.ctg_dim
        gen_adv_loss = tf.reduce_mean(gen_adv_losses)
        gen_loss = gen_adv_loss + hp.enc_w * enc_loss

    hp.gen_opt.minimize(gen_loss, gen.trainable_variables, tape=gen_tape)
    rec_ltn_traces = tf.concat([rec_ltn_traces, rec_ltn_vecs], axis=0)

    hp.ltn_var_trace.assign(hp.ltn_var_trace * hp.ltn_var_decay_rate +
                            tf.reduce_mean(tf.square(rec_ltn_traces), axis=0) * (1.0 - hp.ltn_var_decay_rate))

    fake_scores = tf.reduce_sum(fake_adv_vecs * ctg_vecs, axis=-1) / hp.ctg_dim
    results = {
        'real_adv_val': tf.reduce_mean(real_scores), 'fake_adv_val': tf.reduce_mean(fake_scores),
        'reg_loss': reg_loss, 'enc_loss': enc_loss
    }
    return results


def train(dis: kr.Model, gen: kr.Model, dataset):
    results = {}
    for data in dataset:
        batch_results = _train_step(dis, gen, data)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.convert_to_tensor(results[key]), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    temp_results['ltn_ent'] = hp.get_ltn_ent()
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    return results

