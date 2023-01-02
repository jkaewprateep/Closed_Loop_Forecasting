# Closed_Loop_Forecasting
For study Closed Loop Forecasting in LSTM layer

## LSTM Layer ##

```
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
		self.w = tf.Variable( initial_value=w_init(shape=(input_shape[-1], self.num_units), 
                    dtype='float32'), trainable=True)
		b_init = tf.keras.initializers.Identity( gain=5.0 )
		self.b = tf.Variable( initial_value=b_init(shape=(input_shape[-1], self.num_units), 
                    dtype='float32'), trainable=True)

	def call(self, inputs):

		return tf.matmul(inputs, self.w) + self.b
```

## Results ##

![Closed Loop Forecasting](https://github.com/jkaewprateep/Closed_Loop_Forecasting/blob/main/Figure_15.png "Closed Loop Forecasting")
![Sample 1](https://github.com/jkaewprateep/Closed_Loop_Forecasting/blob/main/Figure_19.png "Sample 1")
![Sample 2](https://github.com/jkaewprateep/Closed_Loop_Forecasting/blob/main/Figure_25.png "Sample 2")

## References ##

1. https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
