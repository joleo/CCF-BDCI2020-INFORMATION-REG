"""
@Time ： 2020/3/26 16:24
@Auth ： joleo
@File ：layers_keras.py
"""


from keras.engine import Layer
import tensorflow as tf
import keras
import keras.backend as K
char_size = 128

class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(char_size, 3, activation='relu', padding='same')
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)


class MaskLSTM(keras.layers.CuDNNLSTM):

    def __init__(self, **kwargs):
        super(MaskLSTM, self).__init__(char_size, return_sequences=True)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskLSTM, self).call(inputs)

class MaskedGlobalMaxPool1D(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs -= K.expand_dims((1.0 - mask) * 1e6, axis=-1)
        return K.max(inputs, axis=-2)


class MaskedGlobalAveragePooling1D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(x, axis=1)

class MaskFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        super(MaskFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        # if mask is not None:
        # mask = K.cast(mask, K.floatx())
        # inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskFlatten, self).call(inputs)  # 调用父类的call ,然后传入inputs


class MaskRepeatVector(keras.layers.RepeatVector):

    def __init__(self, n, **kwargs):
        super(MaskRepeatVector, self).__init__(n, **kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        # if mask is not None:
        # mask = K.cast(mask, K.floatx())
        # inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskRepeatVector, self).call(inputs)


class MaskPermute(keras.layers.Permute):

    def __init__(self, dims, **kwargs):
        super(MaskPermute, self).__init__(dims, **kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        # if mask is not None:
        #     mask = K.cast(mask, K.floatx())
        # inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskPermute, self).call(inputs)

class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    # def compute_mask(self, inputs, mask=None):
    #     return mask
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape

