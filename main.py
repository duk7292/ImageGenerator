import tensorflow as tf
import os
import keras
from keras import layers

import matplotlib.pyplot as plt
import numpy as np
import threading
import time

logger = tf.get_logger()
logger.setLevel('ERROR') 
epochs = 1000000
batch_size = 200


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Create a directory to save the images if it doesn't exist
save_dir = "generated_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_save_dir = "saved_models"

if not os.path.exists(model_save_dir):

    os.makedirs(model_save_dir)


def save_models(epoch, generator_model, discriminator_model, model_save_dir=model_save_dir):

    # Save the generator model
    generator_model_path = os.path.join(model_save_dir, "generator_model_gen_12.tf")
    generator_model.save(generator_model_path, save_format='tf')

    # Save the discriminator model
    discriminator_model_path = os.path.join(model_save_dir, "discriminator_model_gen_12.tf")
    discriminator_model.save(discriminator_model_path, save_format='tf')

    print(f"Models saved for epoch {epoch}")


# Function to load the models
def load_models(model_save_dir=model_save_dir):
    
    # Load the generator model
    generator_model_path = os.path.join(model_save_dir, "generator_model_gen_11.tf")
    generator_model = tf.keras.models.load_model(generator_model_path)

    # Load the discriminator model
    discriminator_model_path = os.path.join(model_save_dir, "discriminator_model_gen_11.tf")
    discriminator_model = tf.keras.models.load_model(discriminator_model_path)

    print("Models loaded")

    return generator_model, discriminator_model



# Pfad zum Ordner mit den Bildern
image_directory = "CatImages/DownScaledCatImages"

# Liste der Bildpfade
image_paths = [os.path.join(image_directory, fname) for fname in os.listdir(image_directory)]
with tf.device('/CPU:0'):
    # Erstellen eines Dataset-Objekts
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Funktion zum Laden und Vorverarbeiten der Bilder
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)  # oder decode_png, falls Ihre Bilder im PNG-Format sind
        image = tf.image.resize(image, [128, 128])
        image /= 255.0  # Normalisierung auf [0,1]
        return image

    # Anwenden der Funktion auf das Dataset
    dataset = dataset.map(load_and_preprocess_image)

    # Batchen und Shuffeln des Datasets
    dataset = dataset.batch(batch_size)  

if os.path.exists(os.path.join(model_save_dir, "generator_model_gen_0.tf")) and os.path.exists(os.path.join(model_save_dir, "discriminator_model_gen_0.tf")):

    generator_model, discriminator_model = load_models()

else:


    generator_model = keras.Sequential([

    keras.layers.InputLayer(input_shape=(100,)),
    
    keras.layers.Dense(8 * 8 * 256),  # Erhöhte Neuronenanzahl
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    
    keras.layers.Dense(8 * 8 * 256),  # Zusätzlicher Dense Layer
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    
    keras.layers.Reshape((8, 8, 256)),
    
    keras.layers.Conv2DTranspose(256, (4, 4), strides=(1, 1), padding='same'),  # Zusätzlicher Conv2DTranspose Layer
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    
    keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    
    keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    
    keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    
    keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')
])


    discriminator_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(128, 128, 3)),
        
        keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        
        keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])


    discriminator_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    generator_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    
    save_models(0, generator_model, discriminator_model)





optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.00001)
for epoch in range(epochs):
    for x,batch in enumerate(dataset):
        real_labels = np.ones((batch.shape[0], 1))
        fake_labels = np.zeros((batch.shape[0], 1))
        noise = np.random.normal(0, 1, (batch.shape[0], 100))
        with tf.GradientTape() as tape:
            # Vorwärtsdurchlauf für echte und gefälschte Bilder
            generated_images = generator_model(noise, training=True)
            logits_real = discriminator_model(batch, training=True)
            logits_fake = discriminator_model(generated_images, training=True)
            
            # Berechnung des Verlustsaber er erkennt d
            loss_real = tf.keras.losses.binary_crossentropy(real_labels, logits_real)
            loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, logits_fake)
            d_loss = 0.5 * (tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake))
        
        # Berechnung der Gradienten und Aktualisierung der Gewichte
        gradients = tape.gradient(d_loss, discriminator_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))
        with tf.GradientTape() as tape_g:
            generated_images = generator_model(noise, training=True)                             
            logits_fake = discriminator_model(generated_images, training=True)
            g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, logits_fake))
        
        gradients_g = tape_g.gradient(g_loss, generator_model.trainable_variables)
        
        optimizer_g.apply_gradients(zip(gradients_g, generator_model.trainable_variables))
        # Berechnung der Genauigkeit

        total_accuracy = 0.5 * (np.mean(logits_real-logits_fake)+1)

        print(f"Epoch {epoch} batch {x}[D loss: {d_loss}, Accuracy: {total_accuracy}]")
        
        if epoch % 1== 0 and x == 0:

            save_models(epoch, generator_model, discriminator_model)

         


        if epoch % 1 == 0 and x == 0:
            image_path = os.path.join(save_dir, f"generated_image_gen_12_epoch_{epoch}_batch{x}.png")
            plt.imshow(generated_images[0])
            plt.axis('off')
            plt.title(f"Epoch {epoch}, Batch {x}")
            plt.savefig(image_path)
            print(f"Image saved at {image_path}")
            plt.close()

