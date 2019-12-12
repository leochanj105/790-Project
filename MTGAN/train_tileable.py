import numpy as np
import os
import tensorflow as tf
import pickle
import random
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

def unpickle(file):
    with open(file, "rb") as f:
        samples = pickle.load(f)
    return samples


def generator(z, kernel_size = 4, reuse = False, lrelu_factor = 0.2, initializer = tf.contrib.layers.xavier_initializer(), training = True):
    with tf.variable_scope('generator', reuse = reuse):
        G_dense = tf.layers.dense(inputs = z, units = 2048)
        G_volumn = tf.reshape(tensor = G_dense, shape = (-1, 2, 2, 512))
        G_h1 = tf.layers.batch_normalization(inputs = G_volumn, training = training)
        G_h1 = tf.maximum(G_h1 * lrelu_factor, G_h1)
            
        G_h2 = tf.layers.conv2d_transpose(filters = 256, strides = 2, kernel_size = kernel_size, 
                                      padding = 'same', inputs = G_h1, activation = None, 
                                      kernel_initializer = initializer)
        G_h2 = tf.layers.batch_normalization(inputs= G_h2, training = training)
        G_h2 = tf.maximum(G_h2 * lrelu_factor, G_h2)
            
        G_h3 = tf.layers.conv2d_transpose(filters = 128, strides = 2, kernel_size = kernel_size, 
                                          padding = 'same', inputs = G_h2, activation = None, 
                                          kernel_initializer = initializer)
        G_h3 = tf.layers.batch_normalization(inputs= G_h3, training = training)
        G_h3 = tf.maximum(G_h3 * lrelu_factor, G_h3)
            
        G_h4 = tf.layers.conv2d_transpose(filters = 64, strides = 2, kernel_size = kernel_size, 
                                          padding = 'same', inputs = G_h3, activation = None, 
                                          kernel_initializer = initializer)
        G_h4 = tf.layers.batch_normalization(inputs= G_h4, training = training)
        G_h4 = tf.maximum(G_h4 * lrelu_factor, G_h4)
            
        G_h5 = tf.layers.conv2d_transpose(filters = 32, strides = 2, kernel_size = kernel_size, 
                                              padding = 'same', inputs = G_h4, activation = None, 
                                              kernel_initializer = initializer)
        G_h5 = tf.layers.batch_normalization(inputs= G_h5, training = training)
        G_h5 = tf.maximum(G_h5 * lrelu_factor, G_h5)
        
#         G_h6 = tf.layers.conv2d_transpose(filters = 64, strides = 2, kernel_size = kernel_size, 
#                                               padding = 'same', inputs = G_h5, activation = None, 
#                                               kernel_initializer = initializer)
#         G_h6 = tf.layers.batch_normalization(inputs= G_h6, training = training)
#         G_h6 = tf.maximum(G_h6 * lrelu_factor, G_h6)
        
#         G_h7 = tf.layers.conv2d_transpose(filters = 32, strides = 2, kernel_size = kernel_size, 
#                                               padding = 'same', inputs = G_h6, activation = None, 
#                                               kernel_initializer = initializer)
#         G_h7 = tf.layers.batch_normalization(inputs= G_h7, training = training)
#         G_h7 = tf.maximum(G_h7 * lrelu_factor, G_h7)
            
        G_logits = tf.layers.conv2d_transpose(filters = 3, strides = 2, kernel_size = kernel_size, 
                                              padding = 'same', inputs = G_h5, activation = None, 
                                              kernel_initializer = initializer)
        result = tf.tanh(x = G_logits)
        return result
    
def discriminator(image, kernel_size = 4, reuse = False, lrelu_factor = 0.2, initializer = tf.contrib.layers.xavier_initializer(), training = True):
    with tf.variable_scope('discriminator', reuse = reuse):
        D_h1 = tf.layers.conv2d(inputs = image, filters = 32, strides = 2, 
                                kernel_size = kernel_size, padding = 'same', 
                                kernel_initializer = initializer)
        D_h1 = tf.maximum(D_h1 * lrelu_factor, D_h1)
        
        D_h2 = tf.layers.conv2d(inputs = D_h1, filters = 64, strides = 2, 
                                kernel_size = kernel_size, padding = 'same', 
                                kernel_initializer = initializer)
        D_h2 = tf.layers.batch_normalization(inputs = D_h2, training = training)
        D_h2 = tf.maximum(D_h2 * lrelu_factor, D_h2)
        
        D_h3 = tf.layers.conv2d(inputs = D_h2, filters = 128, strides = 2, 
                                kernel_size = kernel_size, padding = 'same', 
                                kernel_initializer = initializer)
        D_h3 = tf.layers.batch_normalization(inputs = D_h3, training = training)
        D_h3 = tf.maximum(D_h3 * lrelu_factor, D_h3)
        
        D_h4 = tf.layers.conv2d(inputs = D_h3, filters = 256, strides = 2, 
                                kernel_size = kernel_size, padding = 'same', 
                                kernel_initializer = initializer)
        D_h4 = tf.layers.batch_normalization(inputs = D_h4, training = training)
        D_h4 = tf.maximum(D_h4 * lrelu_factor, D_h4)
        
        D_h5 = tf.layers.conv2d(inputs = D_h4, filters = 512, strides = 2, 
                                kernel_size = kernel_size, padding = 'same', 
                                kernel_initializer = initializer)
        D_h5 = tf.layers.batch_normalization(inputs = D_h5, training = training)
        D_h5 = tf.maximum(D_h5 * lrelu_factor, D_h5)
        
#         D_h6 = tf.layers.conv2d(inputs = D_h5, filters = 1024, strides = 2, 
#                                 kernel_size = kernel_size, padding = 'same', 
#                                 kernel_initializer = initializer)
#         D_h6 = tf.layers.batch_normalization(inputs = D_h6, training = training)
#         D_h6 = tf.maximum(D_h6 * lrelu_factor, D_h6)
        
#         D_h7 = tf.layers.conv2d(inputs = D_h6, filters = 2048, strides = 2, 
#                                 kernel_size = kernel_size, padding = 'same', 
#                                 kernel_initializer = initializer)
#         D_h7 = tf.layers.batch_normalization(inputs = D_h7, training = training)
#         D_h7 = tf.maximum(D_h7 * lrelu_factor, D_h7)
        
        flatten = tf.reshape(tensor = D_h5, shape = (-1, 2048))
        
        D_logits = tf.layers.dense(inputs = flatten, units = 1, activation = None, 
                               kernel_initializer = initializer)
        
        result = tf.sigmoid(x = D_logits)
        return result, D_logits
        
        
        
def setup_training(learning_rate = 0.0005, beta1 = 0.5, beta2 = 0.999, batch_size  = 1, z_dim = 100, width = 64, height = 64):
    tf.reset_default_graph()
    real_images = tf.placeholder(dtype = tf.float32, shape = [batch_size, width, height, 3])
    noise = tf.placeholder(dtype=tf.float32, shape = [batch_size, z_dim], name='noise')
    shamt = tf.placeholder(dtype = tf.int32, shape = (2,), name = 'shamt')
#     fake_images = tf.manip.roll(generator(noise), shift = shamt, axis = [1, 2])
    fake_images = generator(noise)
    fake_images = tf.concat([fake_images[:, height - shamt[0]:,:,:], fake_images[:,:height - shamt[0],:,:]], axis = 1)
    fake_images = tf.concat([fake_images[:,:, width - shamt[1]:,:], fake_images[:,:,:width - shamt[1],:]], axis = 2)
    real_output, real_logits = discriminator(real_images)
    fake_output, fake_logits = discriminator(fake_images, reuse = True)
    G_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = fake_logits, multi_class_labels = tf.ones_like(fake_logits)))
    D_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = real_logits, multi_class_labels = tf.ones_like(real_logits)))
    D_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = fake_logits, multi_class_labels = tf.zeros_like(fake_logits)))
    D_loss = D_loss_real + D_loss_fake
    G_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    D_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        G_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2).minimize(G_loss, var_list = G_vars)
        D_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2).minimize(D_loss, var_list = D_vars)
    return G_optimizer, D_optimizer, G_loss, D_loss, noise, real_images, shamt

def get_part_batch(img, size, batch_size):
    batch = []
    img_w = img.shape[1]
    img_h = img.shape[0]
    width = size[0]
    height = size[1]
    for i in range(batch_size):
        x = random.randint(0, img_w - width) 
        y = random.randint(0, img_h - height)
        batch.append(img[y : y + height,x : x + width , :])
    batch = np.concatenate(batch)
    return batch.reshape((batch_size, height, width, 3)).astype("float") / 255.

def clip_image(image):
    image = np.where(image > 1.0, 1.0, image)
    image = np.where(image < 0.0, 0.0, image)
    return image

def train(batch_size = 64, z_dim = 100, epochs = 101, iters = 10, save_path = "./", image = None):
    G_optimizer, D_optimizer, G_loss, D_loss, noise, real_images, shamt = setup_training(batch_size = batch_size, z_dim = z_dim)
#     image = unpickle("training.pkl")
#     image = image/255.
#     print(image.shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(epochs):
            print("------------------epoch ", epoch, "----------------")
#             shift = [random.randint(0, 64), random.randint(0, 64)]
#             batch = get_part_batch(image, (64, 64), batch_size)
# #             shifted = tf.manip.roll(real_images, shift = shamt, axis = [1, 2])
#             im = tf.concat([real_images[:, 64 - shamt[0]:,:,:], real_images[:,:64 - shamt[0],:,:]], axis = 1)
#             im = tf.concat([im[:,:, 64 - shamt[1]:,:], im[:,:,:64 - shamt[1],:]], axis = 2)
#             real = sess.run(im, feed_dict = {real_images: batch, shamt: shift})[0]
#             plt.imshow(real)
#             plt.show()
            for i in range(iters):
                z = np.random.uniform(-1, 1, [batch_size, z_dim])
                batch = get_part_batch(image, (64, 64), batch_size)
                shift = [random.randint(0, 64), random.randint(0, 64)]
                _ = sess.run(D_optimizer, feed_dict = {noise: z, real_images: batch, shamt: shift})
                
            z = np.random.uniform(-1, 1, [batch_size, z_dim])
            batch = get_part_batch(image, (64, 64), batch_size)
            shift = [random.randint(-16, 16), random.randint(-16, 16)]
            _ = sess.run(G_optimizer, feed_dict = {noise: z, real_images: batch, shamt: shift})
            losses = sess.run([G_loss, D_loss], feed_dict = {noise: z, real_images: batch , shamt: shift})
            print("epoch ", epoch, ", G loss: ", losses[0], ", D loss: ", losses[1])
            z = np.random.uniform(-1, 1, [1, z_dim])
            z_input = tf.placeholder(dtype=tf.float32, shape = [1, z_dim], name='noise')
            generated = sess.run(generator(z_input, reuse = True, training = False), feed_dict = {z_input: z})
            big = np.zeros((512,512,3)).astype("float")
            for i in range(8):
                for j in range(8):
                    big[i*64:(i+1)*64, j*64:(j+1)*64,:] = generated[0]
            plt.imsave(os.path.join(save_path, 'images/image_epoch%d.jpg' % epoch), clip_image(big))
            
            if(epoch % 50 == 0):
                saver.save(sess, os.path.join(save_path, "models/model.ckpt"), global_step = epoch)
                
                
if __name__ == '__main__':
    train_list = []
    with open("images.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            name = line.split("\n")[0]
            train_list.append(name)
            if not os.path.exists("Result/" + name):
                os.mkdir("Result/" + name)
                os.mkdir("Result/" + name + "/images")
                os.mkdir("Result/" + name + "/models")
    for image in train_list:
        print("training: " + image)
        train(save_path = 'Result/' + image + "/", image = unpickle("images/" + image + ".pkl"))
