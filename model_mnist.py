import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

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
        self.n_input = 28 * 28
        self.sample_size = 100
        self.n_size = 28
        self.n_noise = 10
        self.n_hidden = 128
        self.n_channel = 1
        self.n_class = 10
        self.sigma = 0.1
        self.alpha = 0.0000001
        self.grad_clip = 1.0
        self.target_label = 7
        
        self.checkpoint_dir = './checkpoints'
        self.save_file_name = 'mnist_cnn_weight.ckpt'
        
        self.is_training = train
        
        ## Parameter Setting for Classifier
        self.c_weights = {
            'wc1' : tf.Variable(tf.random_normal([5, 5, 1, 32]), name = 'wc1'),
            'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 64]), name = 'wc2'),
            # 7 since 2 times of max pooling (28->14->7)
            'wd1' : tf.Variable(tf.random_normal([7*7*64, 1024]), name = 'wd1'),
            'out' : tf.Variable(tf.random_normal([1024, self.n_class]), name = 'out_w'),
        }

        self.c_biases = {
            'bc1' : tf.Variable(tf.random_normal([32]), name = 'bc1'),
            'bc2' : tf.Variable(tf.random_normal([64]), name = 'bc2'),
            'bd1' : tf.Variable(tf.random_normal([1024]), name = 'bd1'),
            'out' : tf.Variable(tf.random_normal([self.n_class]), name = 'out_b'),
        }
        
        self.C_var_list = [self.c_weights[k] for k in self.c_weights.keys()] + [self.c_biases[k] for k in self.c_biases.keys()]
        
        self.build()
    
    def build(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_input], name = 'gan_X')
        self.classify = self.conv_net(self.X, self.c_weights, self.c_biases, 1)
        self.sess.run(tf.global_variables_initializer())
        # restore variables
        self.saver = tf.train.Saver(var_list = self.C_var_list)
        self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, self.save_file_name))
        
    def attack(self, X, X_tar):
        # Starts with image in target class
        k = 1
        tau = 0.99
        
        for _ in range(10000):
            # backtracking line search
            eps = 0.5 * np.ones_like(X)
            while self.predict(X)[0] < k: 
                # X = self.l_inf_proj(X, X_tar, eps)
                X = self.l_inf_proj2(X, X_tar, eps)
                eps = tau * eps
                
            # projected gradient descent
            noise = anti_sample(self.sample_size, self.n_input)
            adv_images = self.generate(X, noise)
            est_grad = self.estimate_grad(adv_images, noise)

            curr_reward = self.predict(X)[1][self.target_label]
            prev_reward = curr_reward
            while prev_reward <= curr_reward:
                # gradient ascent
                X = X + self.alpha * est_grad

                prev_reward = curr_reward
                curr_reward = self.predict(X)[1][self.target_label]
            
            if _ % 10 == 0:
                print("Query: {}, Epsilon: {}, Rank: {}".format(_, eps[0], self.predict(X)[0]))
                print(curr_reward)
                plt.imshow(np.reshape(X, (28, 28)), cmap=plt.get_cmap('gray'))
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
    
    def conv_net(self, x, weights, biases, dropout):
        x = tf.reshape(x, shape = [-1, 28, 28, 1])

        conv1 = conv2d_t(x, weights['wc1'], biases['bc1'])
        conv1 = maxpool2d(conv1, k=2)

        conv2 = conv2d_t(conv1, weights['wc2'], biases['bc2'])
        conv2 = maxpool2d(conv2, k=2)

        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        #out = tf.nn.softmax(out)

        return out


