import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# LOAD THE TRAINED MODEL
model_path = './trained-model/tumor_model.h5'
if not os.path.exists(model_path):
    print(f"Error: Unable to locate the model file at '{model_path}'.")
    exit()

model = tf.keras.models.load_model(model_path)

# DIRECTORY WITH IMAGES FOR CLASSIFICATION
samples_dir = './samples/'

# LOAD AND PREPARE IMAGES FROM THE DIRECTORY
image_list = []
file_names = []
for filename in os.listdir(samples_dir):
    filepath = os.path.join(samples_dir, filename)
    image = tf.keras.preprocessing.image.load_img(filepath, target_size=(200, 200), color_mode='grayscale')
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_list.append(image_array)
    file_names.append(filename)

# STACK IMAGES INTO A SINGLE NUMPY ARRAY
stacked_images = np.vstack(image_list)

# PREDICT THE CLASSES OF THE IMAGES
class_predictions = model.predict(stacked_images, batch_size=10)

# CLASS LABELS DEFINITION
labels = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# DISPLAY PREDICTIONS
for filename, prediction in zip(file_names, class_predictions):
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = labels[predicted_class_index]
    print(f"{filename}: Predicted Class {predicted_class_index} ({predicted_class_label})")

# VISUALIZE ALL IMAGES IN THE SAMPLES DIRECTORY
total_images = len(file_names)
plt.figure(figsize=(16, 12))
for idx in range(total_images):
    plt.subplot(3, 4, idx + 1)
    plt.imshow(stacked_images[idx].reshape(200, 200), cmap='gray')
    plt.title(f'Predicted: {labels[np.argmax(class_predictions[idx])]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
