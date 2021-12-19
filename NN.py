#This is a modified version of the DQN implementation done by Tejas Kulkarni at  https://github.com/mrkulk/deepQN_tensorflow

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DQN:
    def __init__(self, parameters):
        self.parameters = parameters
        self.net_nme = 'qnet'
        self.sessions = tf.Session()
        self.x = tf.placeholder('float', [None, parameters['width'],parameters['height'], 6],name=self.net_nme + '_x')
        self.q_end_nods = tf.placeholder('float', [None], name=self.net_nme + '_q_end_nods')
        self.actns = tf.placeholder("float", [None, 4], name=self.net_nme + '_actns')
        self.rwrds = tf.placeholder("float", [None], name=self.net_nme + '_rwrds')
        self.end_nods = tf.placeholder("float", [None], name=self.net_nme + '_end_nods')

        #Conv ,Layer 1
        nameLayer = 'conv1' ; size = 3 ; channels = 6 ; filters = 16 ; stride = 1
        self.weight_1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.net_nme + '_'+nameLayer+'_weights')
        self.bias_1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.net_nme + '_'+nameLayer+'_biases')
        self.c1 = tf.nn.conv2d(self.x, self.weight_1, strides=[1, stride, stride, 1], padding='SAME',name=self.net_nme + '_'+nameLayer+'_convs')
        self.output_1 = tf.nn.relu(tf.add(self.c1,self.bias_1),name=self.net_nme + '_'+nameLayer+'_activations')

        #Conv ,Layer 2
        nameLayer = 'conv2' ; size = 3 ; channels = 16 ; filters = 32 ; stride = 1
        self.weight_2 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.net_nme + '_'+nameLayer+'_weights')
        self.bias_2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.net_nme + '_'+nameLayer+'_biases')
        self.model2 = tf.nn.conv2d(self.output_1, self.weight_2, strides=[1, stride, stride, 1], padding='SAME',name=self.net_nme + '_'+nameLayer+'_convs')
        self.output_2 = tf.nn.relu(tf.add(self.model2,self.bias_2),name=self.net_nme + '_'+nameLayer+'_activations')

        output_2_shape = self.output_2.get_shape().as_list()

        #Fully connected 3rd layer
        nameLayer = 'fc3' ; hiddens = 256 ; dim = output_2_shape[1]*output_2_shape[2]*output_2_shape[3]
        self.output_2_flat = tf.reshape(self.output_2, [-1,dim],name=self.net_nme + '_'+nameLayer+'_input_flat')
        self.weight_3 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.net_nme + '_'+nameLayer+'_weights')
        self.bias_3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.net_nme + '_'+nameLayer+'_biases')
        self.input_3 = tf.add(tf.matmul(self.output_2_flat,self.weight_3),self.bias_3,name=self.net_nme + '_'+nameLayer+'_ips')
        self.output_3 = tf.nn.relu(self.input_3,name=self.net_nme + '_'+nameLayer+'_activations')

        #Fully conected 4th layer
        nameLayer = 'fc4' ; hiddens = 4 ; dim = 256
        self.weight_4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.net_nme + '_'+nameLayer+'_weights')
        self.bias_4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.net_nme + '_'+nameLayer+'_biases')
        self.y = tf.add(tf.matmul(self.output_3,self.weight_4),self.bias_4,name=self.net_nme + '_'+nameLayer+'_outputs')


        self.discount = tf.constant(self.parameters['discount'])
        self.yj = tf.add(self.rwrds, tf.multiply(1.0-self.end_nods, tf.multiply(self.discount, self.q_end_nods)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actns), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))

        if self.parameters['load_file'] is not None:
            self.global_step = tf.Variable(int(self.parameters['load_file'].split('_')[-1]),name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)


        self.optim = tf.train.AdamOptimizer(self.parameters['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sessions.run(tf.global_variables_initializer())

        if self.parameters['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sessions,self.parameters['load_file'])


    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_n, self.q_end_nods: np.zeros(bat_n.shape[0]), self.actns: bat_a, self.end_nods:bat_t, self.rwrds: bat_r}
        q_end_nods = self.sessions.run(self.y,feed_dict=feed_dict)
        q_end_nods = np.amax(q_end_nods, axis=1)
        feed_dict={self.x: bat_s, self.q_end_nods: q_end_nods, self.actns: bat_a, self.end_nods:bat_t, self.rwrds: bat_r}
        _,cnt,cost = self.sessions.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sessions, filename)
