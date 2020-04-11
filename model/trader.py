import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
from benchmarking.timer import Timer
#https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
tf.executing_eagerly() 

#tf.compat.v1.enable_eager_execution()
timer = Timer()

#2 define q-network function:
class Model(tf.keras.Model):

    #num_stocks = 500
    
    def __init__(self, num_states, hidden_units, num_stocks):
        print("intialise Keras Model")
        super(Model, self).__init__()
        self.num_stocks = num_stocks
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        #self.num_actions = num_actions
        

        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            1, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        print(z)
        for layer in self.hidden_layers:
            z = layer(z)
            print(z)
        z = self.output_layer(z) 
        output = keras.activations.relu(z, alpha=0, max_value=None, threshold=0)
        #with tf.Session() as sess:  
        #print(output[0])
        return output

class DQN:
    def __init__(self, num_states, num_stocks, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        print("intialise dqn")
        #self.num_actions = num_actions
        self.num_stocks = num_stocks
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = Model(num_states, hidden_units, num_stocks)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        inputs = inputs[0]
        i = 0
        stock_weights = np.empty(shape=len(inputs), dtype=np.float32)
        for inp in inputs:
            for i in range(inp.size):
                if math.isnan(inp[i]):
                    inp[i] = 0

            inp = np.array(inp, dtype=np.float32)
            inp = np.atleast_2d(inp)
            #inp = np.asarray(inp)
            inp = self.model(inp)
            inp = inp.numpy()[0][0]
            #init = tf.compat.v1.global_variables_initializer()
            #with tf.compat.v1.Session() as sess:
                #sess.run(init)
                #inp = sess.run(inp)
                #inp = inp.numpy()
            stock_weights[i] = inp
            i += 1
        #ins = tf.convert_to_tensor(ins, dtype=tf.float32)
        return stock_weights
        #return self.model(np.atleast_2d(inputs.astype('float32')))

    @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_stocks), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            #allocate weights summing up to one
            stock_weights = np.random.dirichlet(np.ones(self.num_stocks), size = 1)[0]
            #print(stock_weights)
            #action = np.random.choice(self.num_stocks, self.num_stocks, )
        else:
            #change to return a predicted weight allocation
            #action = np.argmax(self.predict(np.atleast_2d(states))[0])
            stock_weights = self.predict(np.atleast_2d(states))
        return stock_weights

    '''
    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])
    '''
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

#change function name to "run_trade_session"?
def run_trade_session(market, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    #observations = env.reset()
    observations = market.reset()
    while not done:
        
        actions = TrainNet.get_action(observations, epsilon)
        
        prev_observations = observations
        #timer.start_timer()
        observations, reward, done = market.trade(actions)
        #print("trading " + str(timer.get_time()))
        
        rewards += reward
        if done:
            reward = -200
            market.reset()

        exp = {'s': prev_observations, 'a': actions, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        
    return rewards


def run(market):
    #env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_step = 25
    num_states = market.get_num_states()
    #num_actions = env.action_space.n
    num_stocks = 500
    hidden_units = [10, 10]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_states, num_stocks, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_stocks, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 200
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        print("running trade session: " + str(n))
        total_reward = run_trade_session(market, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards)
    print("avg reward for last 100 episodes:", avg_rewards)
    #make_video(env, TrainNet)
    #market.close()