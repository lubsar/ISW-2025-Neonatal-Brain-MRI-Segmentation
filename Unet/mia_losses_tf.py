import tensorflow as tf
from keras.saving import register_keras_serializable 

# based on https://github.com/JunMa11/SegLossOdyssey/blob/master/losses_pytorch/dice_loss.py
@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = tuple(range(1, len(y_pred.shape))) 
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)

    dice_score = (2. * intersection + smooth) / (denominator + smooth)
    return 1 - tf.reduce_mean(dice_score)

@register_keras_serializable()
def generalized_dice_loss(y_true, y_pred, smooth=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = tuple(range(1, len(y_pred.shape))) 
    w = 1. / (tf.reduce_sum(y_true, axis=axes) ** 2 + smooth)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true + y_pred, axis=axes)

    numerator = 2. * tf.reduce_sum(w * intersection)
    denominator = tf.reduce_sum(w * union)

    return 1 - (numerator + smooth) / (denominator + smooth)

@register_keras_serializable()
def generalized_dice_loss_sparse_labels(y_true, y_pred, smooth=1e-5):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    #one-hot
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape.rank == y_pred.shape.rank else y_true
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)

    axes = tuple(range(1, len(y_pred.shape) - 1))
    w = 1. / (tf.reduce_sum(y_true, axis=axes) ** 2 + smooth)

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true + y_pred, axis=axes)

    numerator = 2. * tf.reduce_sum(w * intersection)
    denominator = tf.reduce_sum(w * union)

    return 1 - (numerator + smooth) / (denominator + smooth)
