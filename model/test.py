from tensorflow.keras import models, layers
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

test_dir = r'C:\Users\Dmitrii\Desktop\model\test'

# This function initializes ImageGenerator for preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=40, #Random rotations
    width_shift_range=0.2, # Random width shift
    height_shift_range=0.2, # Random height shift
    shear_range=0.2, # Shear transformation
    zoom_range=0.2, #
    horizontal_flip=True, #
    fill_mode="nearest" #
)

test_datagen = ImageDataGenerator(rescale=1./255)

# This function loads images from train directory
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Dmitrii\Desktop\model\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# This function loads images from testing directory
validation_generator = test_datagen.flow_from_directory(
    r'C:\Users\Dmitrii\Desktop\model\test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# This function build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)), # Extracts 32 filters (small features) from 3x3 patches of the input image. Output shape is (224, 224, 32)
    layers.MaxPooling2D((2, 2)), # Downsamples the feature map (shrinks spatial size). Output shape is (112, 112, 32)
    layers.Conv2D(64, (3, 3), activation='relu'), # Learns 64 more complex filters
    layers.MaxPooling2D((2, 2)), # Again downsampling
    layers.Conv2D(128, (3, 3), activation='relu'), # Even more filters
    layers.MaxPooling2D((2, 2)), # Shrinks further
    layers.Flatten(), # Converts 3D data to 1D vector
    layers.Dense(512, activation='relu'), # Fully connected hidden layer
    layers.Dense(2, activation='softmax') # For 2 classes, since I have hacker and normal
])

""" Adam: An advanced optimizer (adaptive learning rate, momentum) for faster convergence.

learning_rate=0.001: How big a step to take during training.

loss='categorical_crossentropy':

This is used when I have one-hot encoded labels (e.g., [1, 0] for "hacker", [0, 1] for "normal").

If my labels are just 0 or 1, I'd use 'sparse_categorical_crossentropy' instead. 

metrics=['accuracy']: Track classification accuracy during training. """

model.compile(optimizer=Adam(learning_rate=0.001), 
            loss='categorical_crossentropy', metrics=['accuracy'])

# Using math, calculating actual steps per epoch
train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
val_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

# This function trains the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=val_steps
)

model.save('image_modell.h5')

# Function to loop through test images and predict each
def predict_test_images(test_dir):
    # Looping hrough the subdirectories in the test directory (hacker and norm)
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        
        # Checking if it's a directory and not a file
        if os.path.isdir(class_dir):
            # Looping through each image in the subdirectory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                print(f"Predicting for image: {img_path}")
                predict_image(img_path)  # Call the predict_image function for each image

# Function to predict image class
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)  # Getting model's predictions
    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the class with the highest probability
    class_labels = ['hacker', 'not_hacker']  # Define the class labels
    predicted_label = class_labels[predicted_class]  # Map to the label name

    # Output predictions
    print(f"The model predicts the image is: {predicted_label}")

    # Displaying the image
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.show()

# Calling the function to predict all test images
predict_test_images(test_dir)