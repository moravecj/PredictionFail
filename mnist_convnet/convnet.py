import tensorflow as tf
import tensorflow.contrib.keras as K
import cxflow_tensorflow as cxtf


class SimpleConvNet(cxtf.BaseModel):

    def _create_model(self):
        images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        with tf.variable_scope('conv1'):
            net = K.layers.Conv2D(64, 5)(images)
            net = K.layers.MaxPool2D()(net)
        with tf.variable_scope('conv2'):
            net = K.layers.Conv2D(128, 3)(net)
            net = K.layers.MaxPool2D()(net)
        with tf.variable_scope('dense3'):
            net = K.layers.Flatten()(net)
            net = K.layers.Dropout(0.4).apply(net, training=self.is_training)
            net = K.layers.Dense(64)(net)
        with tf.variable_scope('dense4'):
            logits = K.layers.Dense(10, activation=None)(net)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(logits, 1, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32, name='accuracy'))
