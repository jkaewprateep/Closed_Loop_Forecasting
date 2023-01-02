

import os
from os.path import exists

import tensorflow as tf
import tensorflow_io as tfio

import matplotlib.pyplot as plt


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH = "F:\\temp\\Python\\Speech\\"
FILE_1 = "temple_of_love-sisters_of_mercy.wav"
PATH_FILE_1 = os.path.join(PATH, FILE_1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyLSTMLayer( tf.keras.layers.LSTM ):
	def __init__(self, units, return_sequences, return_state):
		super(MyLSTMLayer, self).__init__( units, return_sequences=True, return_state=False )
		self.num_units = units
		self.w = []
		self.b = []

	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
		shape=[int(input_shape[-1]),
		self.num_units])
		
		w_init = tf.constant_initializer(10.0)
		self.w = tf.Variable( initial_value=w_init(shape=(input_shape[-1], self.num_units), dtype='float32'), trainable=True)
		b_init = tf.keras.initializers.Identity( gain=5.0 )
		self.b = tf.Variable( initial_value=b_init(shape=(input_shape[-1], self.num_units), dtype='float32'), trainable=True)

	def call(self, inputs):

		return tf.matmul(inputs, self.w) + self.b

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
test_file = tf.io.read_file(PATH_FILE_1)	
test_audio, sample_rates = tf.audio.decode_wav(contents=test_file)
sample_rates = tf.constant( sample_rates )

start = 0
limit = 2400
delta = 1
x_scales = tf.range(start=start, limit=limit, delta=delta, dtype=tf.int32, name='range')

layer = MyLSTMLayer( 20, True, False )
layer.build( [1, 20] )


plt.figure(figsize=(3, 1))
plt.suptitle("LSTM Layer results")

plt.subplot(3, 1, 1)
plt.xticks([])
plt.yticks([])
plt.plot( x_scales, test_audio.numpy()[ 15000 : 15000 + limit ] )
# plt.xlabel( "Input as wave mono" )
plt.title( "Input as wave mono" )

start = 0
limit = int( limit / 5 )
delta = 1

x_scales = tf.range(start=start, limit=20 * ( 5 + 1 ), delta=delta, dtype=tf.int32, name='range')

new_input_start = 0
new_input_limit = int( limit / 6 )
new_input_delta = 1

for i in range( 15 ):

	output = []

	for i in range( 5 ) :
		temp = tf.squeeze( layer( tf.expand_dims( test_audio.numpy()[ 15000 + ( limit * i ) : 15000 + ( limit * ( i + 1 ) ) ], axis=1 ) )[0] )
		temp = tf.constant( temp, shape=( 20, ) )
		
		if i > 0 :
			output = tf.concat([ output, temp ], axis=0)
		else :
			output = temp

	
	# new_input = tf.range(start=new_input_start, limit=new_input_limit, delta=new_input_delta, dtype=tf.float32, name='range')
	new_input = tf.random.normal([20], -0.315, 0.315, dtype=tf.float32, name='random_normal')
	
	temp = tf.squeeze( layer( tf.expand_dims( tf.expand_dims( new_input, axis=1 ), axis=2 ) )[0] )
	temp = tf.constant( temp, shape=( 20, ) )
	output = tf.concat([ output, temp ], axis=0)

# tf.random.normal([2,2], 0, 1, tf.float32, seed=1)

plt.subplot(3, 1, 2)
plt.xticks([])
plt.yticks([])
plt.plot( x_scales, output )
# plt.xlabel( "LSTM results" )
plt.title( "LSTM results - 15 rounds " )

plt.margins(0) # remove default margins (matplotlib verision 2+)

plt.axvspan(0, 20, facecolor='papayawhip', alpha=0.5)
plt.axvspan(20, 40, facecolor='white', alpha=0.5)
plt.axvspan(40, 60, facecolor='papayawhip', alpha=0.5)
plt.axvspan(60, 80, facecolor='white', alpha=0.5)
plt.axvspan(80, 100, facecolor='papayawhip', alpha=0.5)
plt.axvspan(100, 120, facecolor='white', alpha=0.5)

for i in range( 20 ):

	output = []

	for i in range( 5 ) :
		temp = tf.squeeze( layer( tf.expand_dims( test_audio.numpy()[ 15000 + ( limit * i ) : 15000 + ( limit * ( i + 1 ) ) ], axis=1 ) )[0] )
		temp = tf.constant( temp, shape=( 20, ) )
		
		if i > 0 :
			output = tf.concat([ output, temp ], axis=0)
		else :
			output = temp

	
	# new_input = tf.range(start=new_input_start, limit=new_input_limit, delta=new_input_delta, dtype=tf.float32, name='range')
	new_input = tf.random.normal([20], -0.5, 0.5, dtype=tf.float32, name='random_normal')
	
	temp = tf.squeeze( layer( tf.expand_dims( tf.expand_dims( new_input, axis=1 ), axis=2 ) )[0] )
	temp = tf.constant( temp, shape=( 20, ) )
	output = tf.concat([ output, temp ], axis=0)
	
plt.subplot(3, 1, 3)
plt.xticks([])
plt.yticks([])
plt.plot( x_scales, output )
# plt.xlabel( "LSTM results" )
plt.title( "LSTM results - 20 rounds" )

plt.margins(0) # remove default margins (matplotlib verision 2+)

plt.axvspan(0, 20, facecolor='papayawhip', alpha=0.5)
plt.axvspan(20, 40, facecolor='white', alpha=0.5)
plt.axvspan(40, 60, facecolor='papayawhip', alpha=0.5)
plt.axvspan(60, 80, facecolor='white', alpha=0.5)
plt.axvspan(80, 100, facecolor='papayawhip', alpha=0.5)
plt.axvspan(100, 120, facecolor='white', alpha=0.5)

plt.show()
