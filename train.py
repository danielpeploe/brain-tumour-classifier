import numpy as np
import cv2
import tensorflow as tf
import os

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from results import plot_training_results, evaluation

# Initialise arrays to store the images and classes
classes = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
x_data = []
y_data = []
img_size = 150

# Function to load and preprocess images
def load_images_from_folder(classes, folder_name):
    for label in classes:
        folder = os.path.join('images', folder_name, label)
        for filename in tqdm(os.listdir(folder), desc=f"Loading images for {label}"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                x_data.append(img)
                y_data.append(label)

# Load training and testing images
load_images_from_folder(classes, 'Training')
load_images_from_folder(classes, 'Testing')
    
# Convert data to numpy arrays and shuffle
x_data, y_data = shuffle(np.array(x_data), np.array(y_data), random_state=101)

# Split the data into training and test sets
train_images, test_images, train_labels, test_labels = train_test_split(x_data, y_data, test_size=0.1, random_state=101)

# Convert class labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical([classes.index(label) for label in train_labels])
test_labels = tf.keras.utils.to_categorical([classes.index(label) for label in test_labels])


# Define the ResNet50 model
resnet = tf.keras.applications.ResNet50(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))

# Build layers on top of ResNet50
x = resnet.output
x = tf.keras.layers.Flatten()(x)  # Flatten the output of ResNet50
x = tf.keras.layers.Dense(1024, activation='relu')(x)  # Add a dense layer with 1024 units
x = tf.keras.layers.Dropout(0.5)(x)  # Add dropout layer with a dropout rate of 0.5
predictions = tf.keras.layers.Dense(4, activation='softmax')(x)  # Final output layer with softmax activation

# Define the model
model = tf.keras.models.Model(inputs=resnet.input, outputs=predictions)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("resnet.keras", monitor="val_loss", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_delta=0.001, mode='auto', verbose=1)

# Train the model
history = model.fit(train_images, train_labels, validation_split=0.1, epochs=12, batch_size=32, verbose=1,
                    callbacks=[checkpoint, reduce_lr])


# Uncomment below lines to produce evaluation graphs
# plot_training_results(history)
evaluation(test_images, test_labels, model)