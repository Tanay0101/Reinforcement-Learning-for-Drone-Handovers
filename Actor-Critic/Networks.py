import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU

class ACTOR_NET:
	def __init__(self, k, NUM_BASE_STATIONS):
		self.k = k
		cells = 3*NUM_BASE_STATIONS
		num_inputs = 2 + 8 + cells
		# Model
		#--------------------------------------------------------------------
		input_A = Input(shape = num_inputs) #(x, y, direction, current_serving_cell_one_hot_encoded)
		x = Dense(32)(input_A)
		x = LeakyReLU()(x)
		x = Dense(64)(x)
		x = LeakyReLU()(x)
		x = Dense(32)(x)
		x = LeakyReLU()(x)
		actor_op = Dense(self.k, activation = 'softmax')(x)

		self.model = Model(inputs = input_A, outputs = actor_op)
		print(self.model.summary())


class CRITIC_NET:
	def __init__(self, k, NUM_BASE_STATIONS):
		self.k = k
		cells = 3*NUM_BASE_STATIONS
		num_inputs = 2 + 8 + cells
		# Model
		#--------------------------------------------------------------------
		input_A = Input(shape = num_inputs) #(x, y, direction, current_serving_cell_one_hot_encoded)
		x = Dense(32)(input_A)
		x = LeakyReLU()(x)
		x = Dense(64)(x)
		x = LeakyReLU()(x)
		x = Dense(32)(x)
		x = LeakyReLU()(x)
		critic_op = Dense(1)(x)

		self.model = Model(inputs = input_A, outputs = critic_op)
		print(self.model.summary())	