import tensorflow as tf

from utils import *
from autoencoder import *

batch_size = 100
batch_shape = (batch_size, 28, 28, 1)
num_visualize = 10

num_epochs = 100

def calculate_loss(original, reconstructed):
    return tf.div(tf.reduce_sum(tf.square(tf.subtract(reconstructed,original))), tf.constant(float(batch_size)))

def train(dataset):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    lr = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.9, staircase=True)
    input_image, reconstructed_image = autoencoder(batch_shape)
    loss = calculate_loss(input_image, reconstructed_image)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        dataset_size = len(dataset.train.images)
        print("Dataset size:", dataset_size)
        num_iters = int((num_epochs * dataset_size)/batch_size)
        print("Num iters:", num_iters)
        for step in range(num_iters):
            input_batch  = get_next_batch(dataset.train, batch_size)
            loss_val,  _ = session.run([loss, optimizer], 
                                       feed_dict={input_image: input_batch})
            if step % 1000 == 0:
                print("Loss at step", step, ":", loss_val)

        test_batch = get_next_batch(dataset.test, batch_size)
        reconstruction = session.run(reconstructed_image,
                                     feed_dict={input_image: test_batch})
        visualize(test_batch, reconstruction, num_visualize)

if __name__ == '__main__':
    dataset = load_dataset()
    train(dataset)
    
