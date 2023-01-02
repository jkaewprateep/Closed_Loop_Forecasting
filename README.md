# Closed_Loop_Forecasting
For study Closed Loop Forecasting in LSTM layer, objectives to predict next output from the input series in scopes and ranges of the training networks and observing change of prediction values from ravirying round loop number and vary data.

## LSTM Layer ##

Simple LSTM layer, with weights initail and bias for training and call() for prediction and training.

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

## Training and Prediction ##

Simply input of 5 different cell, 480 each to LSTM layer with 20 cells and call it once with random input for prediction results.

```
for i in range( 15 ):

    output = []

    for i in range( 5 ) :
        temp = tf.squeeze( layer( tf.expand_dims( test_audio.numpy()[ 15000 + ( limit * i ) : 15000 
					+ ( limit * ( i + 1 ) ) ], axis=1 ) )[0] )
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
```

## Files and Directory ##

| File Name | Description |
--- | --- |
| sample.py | sample codes |
| Figure_15.png | result |
| Figure_19.png | sample data 1 |
| Figure_25.png | sample data 2 |
| README.md | readme file |

## Results ##

![Closed Loop Forecasting](https://github.com/jkaewprateep/Closed_Loop_Forecasting/blob/main/Figure_15.png "Closed Loop Forecasting")
![Sample 1](https://github.com/jkaewprateep/Closed_Loop_Forecasting/blob/main/Figure_19.png "Sample 1")
![Sample 2](https://github.com/jkaewprateep/Closed_Loop_Forecasting/blob/main/Figure_25.png "Sample 2")

## References ##

When read I found examples from this sites, I keep reading to develop my study and games for feedbacks system, forecasting series is a simple task but to have accurate results need some practice see my example from experiences of working with Flappy Birds games and other games applied.

1. https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
