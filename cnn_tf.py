import tensorflow as tf
import numpy as np
import pickle

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 30, 30, 1], name="input")

	conv1 = tf.layers.conv2d(
	  inputs=input_layer,
	  filters=32,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu,
	  name="conv1")
	print("conv1",conv1.shape)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")
	#print("pool1",pool1.shape)

	conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=64,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu,
	  name="conv2")
	#print("conv2",conv2.shape)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3, name="pool2")
	#print("pool2",pool2.shape)

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 5*5*64], name="pool2_flat")
	#print(pool2_flat.shape)
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense")
	#print(dense.shape)
	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")

	# Logits Layer
	num_of_classes = 11
	logits = tf.layers.dense(inputs=dropout, units=num_of_classes, name="logits")

	output_class = tf.argmax(input=logits, axis=1, name="output_class")
	output_probab = tf.nn.softmax(logits, name="softmax_tensor")
	predictions = {"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
	#tf.Print(tf.nn.softmax(logits, name="softmax_tensor"), [tf.nn.softmax(logits, name="softmax_tensor")])
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_of_classes)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)
	#print(len(train_images[1]), len(train_labels))

	classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/cnn_model3")

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_images}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
	classifier.train(input_fn=train_input_fn, steps=500, hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": test_images},
	  y=test_labels,
	  num_epochs=1,
	  shuffle=False)
	test_results = classifier.evaluate(input_fn=eval_input_fn)
	print(test_results)


if __name__ == "__main__":
	tf.app.run()