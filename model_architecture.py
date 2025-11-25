import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable

# --- Custom Activation ---
@register_keras_serializable(package="Custom")
def swish_plus(x):
    """Custom activation function: x * sigmoid(1.5 * x)."""
    return x * tf.nn.sigmoid(1.5 * x)

# --- Squeeze-and-Excitation (SE) Block ---
@register_keras_serializable(package="Custom")
def se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation attention block."""
    filters = input_tensor.shape[-1]
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se) # Reshape to (1, 1, C) for broadcasting
    
    # Excite: Dense layers
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    # Scale input
    return layers.Multiply()([input_tensor, se])

# --- Inception Module ---
@register_keras_serializable(package="Custom")
def inception_module(x, filters):
    """Inception-style module with SE block."""
    branch1 = layers.Conv2D(filters, (1, 1), padding='same', activation=swish_plus)(x)
    branch2 = layers.SeparableConv2D(filters, (3, 3), padding='same', activation=swish_plus)(x)
    branch3 = layers.SeparableConv2D(filters, (5, 5), padding='same', activation=swish_plus)(x)
    branch4 = layers.MaxPooling2D((3, 3), strides=1, padding='same')(x)
    branch4 = layers.Conv2D(filters, (1, 1), padding='same', activation=swish_plus)(branch4)

    x = layers.Concatenate()([branch1, branch2, branch3, branch4])
    x = layers.BatchNormalization()(x)
    x = se_block(x) # Apply attention
    return x

# --- Dynamic Dropout Layer ---
@register_keras_serializable(package="Custom")
class DynamicDropout(layers.Layer):
    """Custom dropout layer with a rate that updates each epoch."""
    def __init__(self, initial_rate=0.1, final_rate=0.5, total_epochs=30, **kwargs):
        super(DynamicDropout, self).__init__(**kwargs)
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_epochs = total_epochs
        self.current_rate = tf.Variable(initial_value=initial_rate, trainable=False, dtype=tf.float32)

    def update_rate(self, epoch):
        """Update the dropout rate based on the current epoch."""
        if epoch < self.total_epochs:
            rate = self.initial_rate + (self.final_rate - self.initial_rate) * (epoch / self.total_epochs)
        else:
            rate = self.final_rate
            
        rate = tf.cast(tf.clip_by_value(rate, self.initial_rate, self.final_rate), tf.float32)
        self.current_rate.assign(rate)

    def call(self, inputs, training=False):
        if training:
            return tf.nn.dropout(inputs, rate=self.current_rate)
        return inputs

    def get_config(self):
        config = super(DynamicDropout, self).get_config()
        config.update({
            "initial_rate": self.initial_rate,
            "final_rate": self.final_rate,
            "total_epochs": self.total_epochs,
        })
        return config

# --- Custom Focal Loss ---
@register_keras_serializable(package="Custom")
class CustomFocalLoss(keras.losses.Loss):
    """Implements Keras-native Focal Loss to address class imbalance."""
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        pt = K.sum(y_true * y_pred, axis=-1)
        focal_component = K.pow(1 - pt, self.gamma)
        loss = focal_component * K.sum(cross_entropy, axis=-1)
        return K.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config

# --- Main Model Builder ---
def build_attentive_lightnet(input_shape=(224, 224, 1), num_classes=2, total_epochs=30):
    input_layer = layers.Input(shape=input_shape)

    # Entry Block
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                      activation=swish_plus,
                      kernel_regularizer=regularizers.l2(0.0001))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Inception Blocks
    x = inception_module(x, 32)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    x = inception_module(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    x = inception_module(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = inception_module(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256, activation=swish_plus, kernel_regularizer=regularizers.l2(0.0001))(x)
    # Note: Updated final_rate to 0.4 based on reference file
    x = DynamicDropout(initial_rate=0.1, final_rate=0.4, total_epochs=total_epochs, name='dynamic_dropout_1')(x)

    x = layers.Dense(128, activation=swish_plus, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = DynamicDropout(initial_rate=0.1, final_rate=0.4, total_epochs=total_epochs, name='dynamic_dropout_2')(x)
    
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer, name="AttentiveLightNet")
    return model