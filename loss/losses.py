# WGAN : https://github.com/henry32144/wgan-gp-tensorflow/blob/master/WGAN-GP-celeb64.ipynb?fbclid=IwAR2QjFjz37Teb7XrawI0jqAwQyGEEPOWLcZXIKibppt_UI2Ce580OjNCxbc
#
"""
image adversarial loss : G, D v
attention loss : G v
conditional expression loss : fake : G, real : D v
Identity loss: G v
"""

import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, CategoricalCrossentropy

def d_loss_fn(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def g_loss_fn(fake_img):
    return -tf.reduce_mean(fake_img)

def gradient_penalty(D, batch_size, real_images, fake_images):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = D(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# def att_loss(attn, lambda_=100):
#     loss = tf.reduce_sum(tf.square(attn[:, :, :-1, :] - attn[:, :, 1:, :])) + \
#            tf.reduce_sum(tf.square(attn[:, :-1, :, :] - attn[:, 1:, :, :]))
#     reg = tf.norm(attn)
#     return lambda_*loss , reg
#
# def identity_loss(gt, pred):
#     mae = MeanAbsoluteError()
#     return mae(gt, pred)
#
# def clf_loss(gt, pred):
#     cross_entropy = CategoricalCrossentropy()
#     return cross_entropy(gt, pred)
#
