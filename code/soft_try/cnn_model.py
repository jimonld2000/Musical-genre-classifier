import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            kernel_initializer="he_normal",
        )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(strides=1, pool_size=2)
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            kernel_initializer="he_normal",
        )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(strides=1, pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer = tf.keras.layers.Dense(
            units=1024, activation=tf.nn.relu, kernel_initializer="he_normal"
        )
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.5)
        self.output_layer = tf.keras.layers.Dense(units=num_classes)
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, training=False):
        x = self.conv_layer_1(inputs)
        x = self.pool_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.pool_layer_2(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        if training:
            x = self.dropout_layer(x, training=training)
        return self.output_layer(x)

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        images, labels = data
        predictions = self(images, training=False)
        loss = self.loss_fn(labels, predictions)
        return {'loss': loss}
