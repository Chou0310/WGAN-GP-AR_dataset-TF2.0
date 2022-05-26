from load_data import *
import tensorflow as tf
from model.wgan import *

#===============================
"""
use  to WGAN GP CK dataset
"""
#===============================
#hyper parameter
EPOCHS = 2000
latent_dim = 128
discriminator_extra_steps = 3
batch_size = 64
gp_weight = 10.0
TRAIN_LOGDIR = './tensor_board/train_WGAN/'
train_img_path = './picture/train_WGAN/'

#load data
LData = data_loader('./classifier_alignment_CK/train/Natural image',
                    './classifier_alignment_CK/train/Expression image',
                    './classifier_alignment_CK/test/Natural image',
                    './classifier_alignment_CK/test/Expression image')
nat2exp_dataset, exp2nat_dataset, total_dataset = LData.build_dataset()

model = WGAN(EPOCHS, latent_dim, discriminator_extra_steps, batch_size, gp_weight, TRAIN_LOGDIR, train_img_path)
model.compile()
model.fit(total_dataset)