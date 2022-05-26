import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_train_img(G, epoch, path='./picture/train_WGAN/'):
    z = tf.random.normal([25, 128])
    img = G(z)
    r, c = 5,   5
    img = tf.reshape(img, shape=(5, 5, 128, 128, 1)).numpy()
    img = img * 127.5 + 127.5
    img = img.astype(np.int32)

    fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(5, 5))
    fig.suptitle('epoch = {}'.format(epoch + 1), fontsize=15)
    for i in range(c):
      for j in range(r):
          axs[i, j].imshow(img[i,j,:,:,:].reshape(128, 128, 1), cmap='gray')
          axs[i, j].axis('off')
    fig.savefig(path + '{}.png'.format(epoch + 1))
    plt.close()

def plot_interpolation_img(G, epoch, path='./picture/interpolation/'):
    z1 = tf.random.normal([10, 1, 128])
    z2 = tf.random.normal([10, 1, 128])
    img1_ratio = np.arange(0, 1, 0.1).reshape((1, 10, 1))
    img2_ratio = 1 - img1_ratio
    z1_list = z1 * img1_ratio
    z2_list = z2 * img2_ratio
    z = z1_list + z2_list
    z = tf.reshape(z, shape=(100, 128))
    img = G(z)
    img = tf.reshape(img, shape=(10, 10, 128, 128, 1)).numpy()
    img = img * 127.5 + 127.5
    img = img.astype(np.int32)
    r, c = 10, 10
    fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(5, 5))
    fig.suptitle('epoch = {}'.format(epoch + 1), fontsize=15)
    for i in range(c):
      for j in range(r):
          axs[i, j].imshow(img[i,j,:,:,:].reshape(128, 128, 1), cmap='gray')
          axs[i, j].axis('off')
    fig.savefig(path + '{}.png'.format(epoch + 1))
    plt.close()

def save_interpolation_img(G, epoch, path='./picture/interpolation dataset/'):
    z1 = tf.random.normal([5, 1, 128])
    z2 = tf.random.normal([5, 1, 128])
    img1_ratio = np.arange(0, 1, 0.2).reshape((1, 5, 1))
    img2_ratio = 1 - img1_ratio
    z1_list = z1 * img1_ratio
    z2_list = z2 * img2_ratio
    z = z1_list + z2_list
    z = tf.reshape(z, shape=(25, 128))
    img = G(z)
    img = img * 127.5 + 127.5
    img = img.numpy()

    img = img.astype(np.int32)
    for i in range(img.shape[0]):
        cv2.imwrite(path + str(i) + '.png', img[i].reshape(128, 128, 1))

    r, c = 5, 5
    fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(5, 5))
    fig.suptitle('epoch = {}'.format(epoch + 1), fontsize=15)
    img = img.reshape(5, 5, 128, 128, 1)
    for i in range(c):
      for j in range(r):
          axs[i, j].imshow(img[i,j,:,:,:].reshape(128, 128, 1), cmap='gray')
          axs[i, j].axis('off')
    fig.savefig(path + 'total.png')
    plt.close()

    # z1 = tf.random.normal([5, 1, 128])
    # z2 = tf.random.normal([5, 1, 128])
    # img1_ratio = np.arange(0, 1, 0.2).reshape((1, 5, 1))
    # img2_ratio = 1 - img1_ratio
    # z1_list = z1 * img1_ratio
    # z2_list = z2 * img2_ratio
    # z = z1_list + z2_list
    # z = tf.reshape(z, shape=(25, 128))
    # img = G(z)
    # img = tf.reshape(img, shape=(5, 5, 128, 128, 1)).numpy()
    # img = img * 127.5 + 127.5
    # img = img.astype(np.int32)
    # r, c = 5, 5
    # fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(5, 5))
    # fig.suptitle('epoch = {}'.format(epoch + 1), fontsize=15)
    # for i in range(c):
    #   for j in range(r):
    #       axs[i, j].imshow(img[i,j,:,:,:].reshape(128, 128, 1), cmap='gray')
    #       axs[i, j].axis('off')
    # fig.savefig(path + '{}.png'.format(epoch + 1))
    # plt.close()