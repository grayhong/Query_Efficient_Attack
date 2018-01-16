import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from vgg.vgg16trunc import vgg
from scipy.misc import imread, imresize
from ops import conv2d_t, lrelu, deconv2d, fully_connect, conv_cond_concat, batch_normal, squash, maxpool2d
from utils import anti_sample, softmax
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Attacker(object):
    
    def __init__(self, sess, train=True, target_label = 0):

        # self.test_x, self.test_y, self.test_l = get_data_set("test", cifar=10)
        self.sess = sess
        
        # Parameters
        self.total_epoch = 100
        self.learning_rate = 5e-4
        self.batch_size = 100
        self.sample_size = 10
        self.n_input = 224 * 224 * 3
        self.n_size = 224
        self.n_channel = 3
        self.n_noise = 10
        self.n_hidden = 128
        self.n_class = 1000
        self.sigma = 0.1
        self.alpha = 0.1
        self.grad_clip = 1.0
        self.target_label = target_label
        
        self.checkpoint_dir = "./cifar10/tensorboard/cifar-10/"
        
        self.is_training = train
        
        self.build()
    
    def build(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_input], name = 'gan_X')
        self.Y = tf.placeholder(tf.float32, [None, self.n_class], name='cnn_Y')
        self.x_image = tf.reshape(self.X, [-1, self.n_size, self.n_size, self.n_channel], name='images')
        self.classify, self.vgg_params = vgg(self.x_image, reuse=False)
        
        self.sess.run(tf.global_variables_initializer())
        # restore variables
        weights = np.load('./vgg/vgg16_weights.npz')
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            self.sess.run(self.vgg_params[i].assign(weights[k]))
            
        
    def attack(self, X, X_tar):
        # Starts with image in target class
        k = 2
        tau = 0.95
        eps = 1 * np.ones_like(X)
        
        for _ in range(10000):
            # backtracking line search
            while self.predict(X)[0] < k: 
                # X = self.l_inf_proj(X, X_tar, eps)
                X = self.l_inf_proj(X, X_tar, eps)
                eps = tau * eps
                
            # projected gradient descent
            noise = anti_sample(self.sample_size, self.n_input)
            adv_images = self.generate(X, noise)
            est_grad = self.estimate_grad(adv_images, noise)

            curr_reward = self.predict(X)[1][self.target_label]
            prev_reward = curr_reward
            while prev_reward <= curr_reward:
                # gradient ascent
                X = X + self.alpha * eps[0] * est_grad
                X = np.clip(X, 0, 1)

                prev_reward = curr_reward
                curr_reward = self.predict(X)[1][self.target_label]
            
            if _ % 1 == 0:
                print("Query: {}, Epsilon: {}, Rank: {}".format(_, eps[0], self.predict(X)[0]))
                print(curr_reward)
                plt.imshow(np.reshape(X, (32, 32, 3)), cmap=plt.get_cmap('gray'))
                plt.show()
    
    def predict(self, X):
        if len(X.shape) > 1:
            batch_size = X.shape[0]
            output = self.sess.run(self.classify, feed_dict={self.X: X})
            # output = [softmax(t) for t in output]
            rank = np.where(np.argsort(output) == self.target_label)[0][0]
        else:
            output = self.sess.run(self.classify, feed_dict={self.X: [X]})[0]
            # output = softmax(output)
            rank = 9 - np.where(np.argsort(output) == self.target_label)[0][0]
        return rank, output
        
    def generate(self, X, noise):
        theta = X + self.sigma * noise
        return theta
    
    def estimate_grad(self, adv_images, noise):
        # N x 1
        _, output = self.predict(adv_images)
        target_arr = np.array([self.target_label] * self.sample_size)
        # print(reward)
        reward = output[np.arange(self.sample_size), target_arr]
        
        # N x D
        per_reward = (noise.T * reward).T
        mean_reward = np.mean(per_reward, 0)
        grad = 1 / self.sigma * mean_reward
        return grad
    
    def l_inf_proj(self, X_adv, X_obj, eps):
        # +1 if X_target > X else, -1 (for each pixel)
        sign = np.sign(X_adv - X_obj)
        perturb = np.minimum.reduce([np.abs(X_adv - X_obj), eps])
        apply = X_obj + sign * perturb
        return apply
    
    def l_inf_proj2(self, X_adv, X_obj, eps):
        # +1 if X_target > X else, -1 (for each pixel)
        sign = np.sign(X_adv - X_obj)
        amp = np.sum(np.abs(X_adv - X_obj))
        eps_amp = eps[0]
        if amp > eps_amp:
            perturb = (X_adv - X_obj) * eps_amp / amp
        else:
            perturb = (X_adv - X_obj)
            
        apply = X_obj + sign * perturb
        return apply
