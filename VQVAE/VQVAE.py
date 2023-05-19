import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from ResNetAE import ResNetEncoder, ResNetDecoder


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


def get_encoder(latent_dim=16):
    return ResNetEncoder(z_dim=latent_dim, name='encoder')


def get_decoder(latent_dim=16):
    return ResNetDecoder(output_channels=1, name='decoder')


def get_vqvae(latent_dim=16, num_embeddings=64):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(256, 256, 1))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


def get_vqvae2(latent_dim=16, num_embeddings=64):
    vq_layer_t = VectorQuantizer(num_embeddings, num_embeddings, name="vector_quantizer_t")
    vq_layer_b = VectorQuantizer(num_embeddings, num_embeddings, name="vector_quantizer_b")
    quantize_conv_t = tf.keras.layers.Conv2D(filters=num_embeddings,  # SUPPOSED TO BE 'latent_dim'
                                             kernel_size=(1, 1), strides=(1, 1), padding='same')
    quantize_conv_b = tf.keras.layers.Conv2D(filters=num_embeddings,  # SUPPOSED TO BE 'latent_dim'
                                             kernel_size=(1, 1), strides=(1, 1), padding='same')
    upsample_layer = tf.keras.layers.Conv2DTranspose(num_embeddings,  # SUPPOSED TO BE 'latent_dim'
                                                     kernel_size=(8, 8), strides=(4, 4), padding='same')
    encoder_t = ResNetEncoder(z_dim=latent_dim, n_levels=2, name='encoder_t')
    encoder_b = ResNetEncoder(z_dim=latent_dim, n_levels=2, name='encoder_b')
    decoder_t = ResNetDecoder(output_channels=latent_dim, n_levels=2, name='decoder_t')
    decoder_b = ResNetDecoder(output_channels=1, n_levels=2, name='decoder_b')

    inputs = keras.Input(shape=(256, 256, 1))

    enc_b = encoder_b(inputs)
    enc_t = encoder_t(enc_b)
    quant_t = vq_layer_t(quantize_conv_t(enc_t))

    dec_t = decoder_t(quant_t)
    enc_b = tf.concat([dec_t, enc_b], axis=-1)
    quant_b = vq_layer_b(quantize_conv_b(enc_b))

    upsample_t = upsample_layer(quant_t)
    quantized_latents = tf.concat([upsample_t, quant_b], axis=-1)
    reconstructions = decoder_b(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae_2")


# get_vqvae().summary()


class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        # self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)
        self.vqvae = get_vqvae2(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def call(self, x):
        return self.vqvae(x)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
        
    def get_model(self):
        return self.vqvae
