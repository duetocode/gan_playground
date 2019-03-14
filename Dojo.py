import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.keras import backend as K

from models import build_discriminator, build_generator

class Dojo():

    def __init__(self, training_ratio=5):
        self.training_ratio = training_ratio
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        self.optimizer_geneator = AdamOptimizer(1e-4, beta1=0.5)
        self.optimizer_discriminator = AdamOptimizer(1e-4, beta1=0.5)

    def train_on_batch(self, steps, images):
        loss_d, loss_g = None, None
        with tf.GradientTape() as tape_generator, tf.GradientTape() as tape_discriminator:
            z = K.random_uniform((K.int_shape(images)[0], 512))
            generated = self.generator(z, training=True)
            logits_generated = self.discriminator(generated, training=True)
            logits_real = self.discriminator(images, training=True)
            loss_d = wasserstein_gp_loss(logits_generated, logits_real, generated, images, self.discriminator)

            if steps % self.training_ratio == 0:
                loss_g = -1.0 * K.mean(logits_generated)
            
        _update(loss_d, self.discriminator, self.optimizer_discriminator, tape_discriminator)
        _update(loss_g, self.generator, self.optimizer_geneator, tape_generator)

        return loss_g, loss_d
    
    def run(self, batch_size=4):
        z = K.random_uniform((batch_size, 512))
        generated = self.generator(z, training=False)

        return generated


def _update(loss, model, optimizer, tape):
    if loss is None:
        return
    
    gradients = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(gradients, model.variables))

def wasserstein_gp_loss(logits_generated, logits_real, generated, real, discriminator):

    # Gradient Penalty
    alpha = K.random_uniform(
                            shape = (K.int_shape(generated)[0], 1, 1, 1),
                            minval=0.0,
                            maxval=1.0)
    differences = generated - real
    interpolates = real + (alpha * differences)
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        logits_interpolates = discriminator(interpolates, training=True)

    gradients = tape.gradient(logits_interpolates, interpolates)[0]
    slopes = K.sqrt(K.sum(K.square(gradients), axis=[1]))
    gradient_penalty = K.mean(K.square(slopes-1.))

    loss_D = K.mean(logits_generated) - K.mean(logits_real) + 10.0 * gradient_penalty
    return loss_D
    