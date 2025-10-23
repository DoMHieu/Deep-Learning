import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

class Attention(Layer):
    def call(self, inputs):
        q = tf.nn.tanh(inputs)
        score = tf.nn.softmax(tf.reduce_sum(q, axis=2, keepdims=True), axis=1)
        context = tf.reduce_sum(score * inputs, axis=1)
        return context
