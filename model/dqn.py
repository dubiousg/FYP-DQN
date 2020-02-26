env = gym.make("MsPacman-v0")
obs = env.reset()
obs.shape # [height, width, channels]
(210, 160, 3)
env.action_space

mspacman_color = np.array([210, 164, 74]).mean()
def preprocess_observation(obs):
img = obs[1:176:2, ::2] # crop and downsize
img = img.mean(axis=2) # to greyscale
img[img==mspacman_color] = 0 # improve contrast
img = (img - 128) / 128 - 1 # normalize from -1. to 1.
return img.reshape(88, 80, 1)


input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10 # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()
def q_network(X_state, name):
prev_layer = X_state
conv_layers = []
with tf.variable_scope(name) as scope:
for n_maps, kernel_size, stride, padding, activation in zip(
conv_n_maps, conv_kernel_sizes, conv_strides,
conv_paddings, conv_activation):
prev_layer = tf.layers.conv2d(
prev_layer, filters=n_maps, kernel_size=kernel_size,
stride=stride, padding=padding, activation=activation,
kernel_initializer=initializer)
conv_layers.append(prev_layer)
last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
activation=hidden_activation,
kernel_initializer=initializer)
outputs = tf.layers.dense(hidden, n_outputs,
kernel_initializer=initializer)
trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
scope=scope.name)
trainable_vars_by_name = {var.name[len(scope.name):]: var
for var in trainable_vars}
return outputs, trainable_vars_by_name
