import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3,1), activation=tf.nn.relu, padding='same')
        # self.maxp1 = tf.keras.layers.MaxPooling2D(, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3,1), padding='same', activation=tf.nn.relu)
        # self.maxp2 = tf.keras.layers.MaxPooling2D(2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=(3,1), padding='same', activation=tf.nn.relu)

    def call(self, inputs):
        act = self.conv1(inputs)
        # act = self.maxp1(act)
        act = self.conv2(act)
        # act = self.maxp2(act)
        act = self.conv3(act)
        # mu = self.mu(act)
        # sigma = self.sigma(act)
        return act


class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        # self.upsamp1 = tf.keras.layers.UpSampling2D(2)
        self.conv5 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), activation=tf.nn.relu, padding='same')
        # self.upsamp2 = tf.keras.layers.UpSampling2D(2)
        self.conv6 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), activation=tf.nn.relu, padding='same')
        # self.upsamp3 = tf.keras.layers.UpSampling2D(1)
        self.conv7 = tf.keras.layers.Conv2DTranspose(784, kernel_size=(3,3), activation='linear', padding='same')

    def call(self, inputs):
        act = self.conv4(inputs)
        # act = self.upsamp1(act)
        act = self.conv5(act)
        # act = self.upsamp2(act)
        act = self.conv6(act)
        # act = self.upsamp3(act)
        return self.conv7(act)


class ConvAE(tf.keras.Model):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def call(self, inputs):
        act = self.encoder(inputs)
        return self.decoder(act)

    def train_model(self, data, bsize, epochs=2, lr=0.012):
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
        self.fit(data, data, batch_size=bsize, epochs=epochs, shuffle=True)