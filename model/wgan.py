import time
import os
from loss.losses import *
from network.pomelo_network import *
from utils.img_tools import *
from network import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import imageio

class WGAN(Model):
    def __init__(self,
        EPOCH = 100,
        latent_dim=128,
        discriminator_extra_steps=3,
        batch_size = 64,
        gp_weight=10.0,
        TRAIN_LOGDIR=None,
        train_img_path=None
    ):
        super(WGAN, self).__init__()
        self.EPOCHS = EPOCH
        self.discriminator = get_discriminator_model()
        self.generator = get_generator_model()
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.batch_size = batch_size
        self.gp_weight = gp_weight
        self.TRAIN_LOGDIR = TRAIN_LOGDIR
        self.train_img_path = train_img_path
        # #tensorboard
        self.train_G_writer = tf.summary.create_file_writer(self.TRAIN_LOGDIR + 'G/')
        self.train_D_writer = tf.summary.create_file_writer(self.TRAIN_LOGDIR + 'D/')

    def compile(self):
        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    def fit(self, train_dataset):
        #checkpoint
        ckpt = tf.train.Checkpoint(generator=self.generator,
                                   discriminator=self.discriminator,
                                   G_optimizer=self.g_optimizer,
                                   D_optimizer=self.d_optimizer)
        manager = tf.train.CheckpointManager(ckpt, directory='./checkpoint/train_WGAN', checkpoint_name='model.ckpt', max_to_keep=10)

        #training
        for epoch in range(self.EPOCHS):
            print("\nepoch %d/%d " % (epoch + 1, self.EPOCHS))
            start = time.time()
            # Iterate over the batches of the dataset.
            for step, data in enumerate(train_dataset):
                real_img, nat_label, exp_label, exp_img = data
                real_img = tf.convert_to_tensor(self.get_image(real_img))
                for i in range(self.d_steps):
                    D_adv_loss = self.train_D_step(real_img)
                G_adv_loss = self.train_G_step(real_img)

                if step % 4 == 0:
                    print('.', end='')


            if epoch % 2 == 0:
                manager.save()

            if epoch % 10 == 0:
                plot_train_img(self.generator, epoch, path=self.train_img_path)

            #tensorboard record
            with self.train_G_writer.as_default():
                tf.summary.scalar('adv_loss', G_adv_loss, step=epoch)

            with self.train_D_writer.as_default():
                tf.summary.scalar('adv_loss', D_adv_loss, step=epoch)

            end = time.time()
            print("\n this epoch costing time = {:.2f}".format(end - start), end='')

        self.save_gif(self.train_img_path)




    @tf.function
    def train_G_step(self, real_images):
        random_latent_vectors = tf.random.normal([real_images.shape[0], self.latent_dim])
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return g_loss

    @tf.function
    def train_D_step(self, real_images):
        random_latent_vectors = tf.random.normal([real_images.shape[0], self.latent_dim])
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator(random_latent_vectors, training=True)
            # Get the logits for the fake images
            fake_logits = self.discriminator(fake_images, training=True)
            # Get the logits for the real images
            real_logits = self.discriminator(real_images, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = d_loss_fn(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = gradient_penalty(self.discriminator, real_images.shape[0], real_images, fake_images)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * self.gp_weight

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
        return d_loss

    def get_image(self, path):
        total_img =[self.read_img(x) for x in path.numpy()]
        total_img = tf.image.resize(total_img, [128, 128])
        return total_img

    def read_img(self, path):
        img = tf.cast(tf.io.decode_png(open(path.decode('UTF-8'), 'rb').read(), channels=1), dtype=tf.float32)
        img = (img - 127.5)/127.5
        return img

    def save_gif(self, path):
        images = []
        filenames = os.listdir(path)
        for filename in sorted(filenames):
            print(filename)
            images.append(imageio.imread(path + filename))
        imageio.mimsave('./gif/train_img.gif', images, fps=10)

