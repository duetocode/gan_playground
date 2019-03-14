import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import backend as K


def _build(self, input_shape):
  self.u = self.add_weight('u', 
                  [1, K.int_shape(self.kernel)[-1]], 
                  initializer='random_normal', 
                  trainable=False)

def _spectral_norm(self, call_method, inputs, **kwargs):
  w_shape = K.int_shape(self.kernel)
  w = tf.reshape(self.kernel, [-1, w_shape[-1]])

  u = self.u

  u_hat  = self.u
  v_hat  = None

  for _ in range(1):
    """
    power iteration
    Usually iteration = 1 will be enough
    """
    v_ = K.dot(u_hat, tf.transpose(w))
    v_hat = K.l2_normalize(v_)

    u_ = K.dot(v_hat, w)
    u_hat = K.l2_normalize(u_)

  u_hat = K.stop_gradient(u_hat)
  v_hat = K.stop_gradient(v_hat)

  sigma = K.dot(K.dot(v_hat, w), K.transpose(u_hat))
  with tf.control_dependencies([u.assign(u_hat)]):
    w_norm = w / sigma
    w_norm = K.reshape(w_norm, w_shape)

  kernel = self.kernel
  self.kernel = w_norm
  result = call_method(inputs, *kwargs)
  self.kernel = kernel
  return result

class Conv2D(layers.Conv2D):
    def build(self, input_shape):
      super(Conv2D, self).build(input_shape)
      _build(self, input_shape)

    def call(self, inputs, **kwargs):
      return _spectral_norm(self, super(Conv2D, self).call, inputs, *kwargs)

class Dense(layers.Dense):
    def build(self, input_shape):
      super(Dense, self).build(input_shape)
      _build(self, input_shape)

    def call(self, inputs, **kwargs):
      return _spectral_norm(self, super(Dense, self).call, inputs, *kwargs)

class Conv2DTranspose(layers.Conv2DTranspose):
    def build(self, input_shape):
      super(Conv2DTranspose, self).build(input_shape)
      _build(self, input_shape)

    def call(self, inputs, **kwargs):
      return _spectral_norm(self, super(Conv2DTranspose, self).call, inputs, *kwargs)