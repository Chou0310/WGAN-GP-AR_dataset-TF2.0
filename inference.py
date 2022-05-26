from load_data import *
import tensorflow as tf
from model.wgan import *
from utils.img_tools import *


#hyper parameter
EPOCHS = 200
latent_dim = 128
discriminator_extra_steps = 3
batch_size = 64
gp_weight = 10.0
TRAIN_LOGDIR = './tensor_board/train_WGAN/'
test_img_path = './picture/interpolation/'

model = WGAN(EPOCHS, latent_dim, discriminator_extra_steps, batch_size, gp_weight, TRAIN_LOGDIR, test_img_path)
model.compile()
ckpt = tf.train.Checkpoint(generator=model.generator,
                           discriminator=model.discriminator,
                           G_optimizer=model.g_optimizer,
                           D_optimizer=model.d_optimizer)
ckpt.restore(tf.train.latest_checkpoint('./checkpoint/train_WGAN'))  # 从文件恢复模型参数

for epoch in range(10):
    plot_interpolation_img(model.generator, epoch)
# epoch = 2000
# save_interpolation_img(model.generator, epoch)
#interpolation S2S NS2NS S2NS NS2S