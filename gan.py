import os
import time
from matplotlib import pyplot as plt
import tensorflow as tf

from keras.utils import img_to_array
from keras.utils import load_img
import numpy as np
from keras import layers
from keras import models
from keras.optimizers import Adam

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


dir_real_faces = "data/img_align_celeba"
dir_fake_faces = "ai_generated_faces"
img_shape = (32, 32, 3)

def get_npdata(dir_real, dir_fake, img_shape, max_images=900):
    X_real = []
    X_fake = []

    real_images = sorted(os.listdir(dir_real))
    fake_images = sorted(os.listdir(dir_fake))

    for img_name in real_images[:max_images]:
        image = load_img(os.path.join(dir_real, img_name), target_size=img_shape[:2])
        image = img_to_array(image) / 255.0
        X_real.append(image)

    for img_name in fake_images[:max_images]:
        image = load_img(os.path.join(dir_fake, img_name), target_size=img_shape[:2])
        image = img_to_array(image) / 255.0
        X_fake.append(image)

    X_real = np.array(X_real)
    X_fake = np.array(X_fake)

    return X_real, X_fake


# def get_npdata(dir_real, dir_fake, img_shape):
#     X_real = []
#     X_fake = []

#     real_images = sorted(os.listdir(dir_real))
#     fake_images = sorted(os.listdir(dir_fake))

#     for img_name in real_images:
#         image = load_img(os.path.join(dir_real, img_name), target_size=img_shape[:2])
#         image = img_to_array(image) / 255.0
#         X_real.append(image)

#     for img_name in fake_images:
#         image = load_img(os.path.join(dir_fake, img_name), target_size=img_shape[:2])
#         image = img_to_array(image) / 255.0
#         X_fake.append(image)

#     X_real = np.array(X_real)
#     X_fake = np.array(X_fake)

#     return X_real, X_fake

X_real, X_fake = get_npdata(dir_real_faces, dir_fake_faces, img_shape)
print("X_real.shape =", X_real.shape)
print("X_fake.shape =", X_fake.shape)

optimizer = Adam(0.00007, 0.5)
noise_shape = (100,)

def build_generator(img_shape, noise_shape=(100,)):
    input_noise = layers.Input(shape=noise_shape)
    d = layers.Dense(1024, activation="relu")(input_noise)
    d = layers.Dense(1024, activation="relu")(d)
    d = layers.Dense(128 * 8 * 8, activation="relu")(d)
    d = layers.Reshape((8, 8, 128))(d)

    d = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
    d = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name="block_4")(d)

    d = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
    d = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name="block_5")(d)

    if img_shape[0] == 64:
        d = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
        d = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name="block_6")(d)

    img = layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same', name="final_block")(d)
    model = models.Model(input_noise, img)
    model.summary()
    return model

generator = build_generator(img_shape, noise_shape=noise_shape)
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

def build_discriminator(img_shape, noutput=1):
    input_img = layers.Input(shape=img_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(noutput, activation='sigmoid')(x)
    model = models.Model(input_img, out)

    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator.summary()

def train_discriminator(discriminator, X_real, X_fake, epochs=1000, batch_size=128):
    half_batch = int(batch_size / 2)
    history = []

    for epoch in range(epochs):
        idx_real = np.random.randint(0, X_real.shape[0], half_batch)
        idx_fake = np.random.randint(0, X_fake.shape[0], half_batch)

        real_imgs = X_real[idx_real]
        fake_imgs = X_fake[idx_fake]

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        history.append(d_loss)

        if epoch % 100 == 0:
            print("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%]".format(epoch, d_loss[0], 100 * d_loss[1]))

    return history

discriminator_history = train_discriminator(discriminator, X_real, X_fake)

z = layers.Input(shape=noise_shape)
img = generator(z)

discriminator.trainable = False
valid = discriminator(img)

combined = models.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)
combined.summary()

def get_noise(nsample=1, nlatent_dim=100):
    noise = np.random.normal(0, 1, (nsample, nlatent_dim))
    return noise

def plot_generated_images(noise, path_save=None, titleadd=""):
    imgs = generator.predict(noise)
    fig = plt.figure(figsize=(40, 10))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i + 1)
        ax.imshow(img)
    fig.suptitle("Generated images " + titleadd, fontsize=30)

    if path_save is not None:
        plt.savefig(path_save, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def train(models, X_train, noise_plot, dir_result="./result/", epochs=10000, batch_size=128):
    combined, discriminator, generator = models
    nlatent_dim = noise_plot.shape[1]
    half_batch = int(batch_size / 2)
    history = []
    for epoch in range(epochs):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        noise = get_noise(half_batch, nlatent_dim)

        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = get_noise(batch_size, nlatent_dim)

        valid_y = np.array([1] * batch_size).reshape(batch_size, 1)

        g_loss = combined.train_on_batch(noise, valid_y)

        history.append({"D": d_loss[0], "G": g_loss})

        if epoch % 100 == 0:
            print("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%] [G loss: {:4.3f}]".format(
                epoch, d_loss[0], 100 * d_loss[1], g_loss))
        if epoch % int(epochs / 100) == 0:
            plot_generated_images(noise_plot,
                                  path_save=dir_result + "/image_{:05.0f}.png".format(epoch),
                                  titleadd="Epoch {}".format(epoch))
        # if epoch % 1000 == 0:
        #     plot_generated_images(noise_plot,
        #                           titleadd="Epoch {}".format(epoch))

    return history

dir_result = "./result_GAN/"

try:
    os.mkdir(dir_result)
except FileExistsError:
    pass

start_time = time.time()

_models = combined, discriminator, generator

history = train(_models, X_real, get_noise(nsample=4, nlatent_dim=noise_shape[0]), dir_result=dir_result, epochs=500, batch_size=128 * 8)
end_time = time.time()
print("-" * 10)
print("Time took: {:4.2f} min".format((end_time - start_time) / 60))

def evaluate_frames(discriminator, frame_paths, img_shape):
    images = []
    for frame_path in frame_paths:
        image = load_img(frame_path, target_size=img_shape[:2])
        image = img_to_array(image) / 255.0
        images.append(image)
    images = np.array(images)
    predictions = discriminator.predict(images)
    return predictions

video_frames_dir = "extracted_faces"
frame_paths = [os.path.join(video_frames_dir, fname) for fname in os.listdir(video_frames_dir) if fname.endswith('.png')]
predictions = evaluate_frames(discriminator, frame_paths, img_shape)
for i, prediction in enumerate(predictions):
    print(f"Image  AI gener{i}ated: {prediction[0] > 0.5}")