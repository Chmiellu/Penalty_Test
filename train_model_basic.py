from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# TRAIN? 1, TEST? 0
TRAIN = 1

training_data_dir = "data/training"
validation_data_dir = "data/validation"
test_data_dir = "data/test"
MODEL_SUMMARY_FILE = "model_summary_basic.txt"

# hyperparameters
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128
RGB = 3
EPOCHS = 10
BATCH_SIZE = 4

# create BASIC model
model = Sequential()

# Input layer + one hidden layer
model.add(Flatten(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, RGB)))
model.add(Dense(64, activation='sigmoid'))

# Output layer
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0005), metrics=['accuracy'])

model.summary()

with open(MODEL_SUMMARY_FILE, "w", encoding="utf-8") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

# data augmentation
training_data_generator = ImageDataGenerator(
    rescale=1/128,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=5,
    horizontal_flip=False)

validation_data_generator = ImageDataGenerator(
    rescale=1/128
)

test_data_generator = ImageDataGenerator(
    rescale=1/128
)

# load data using flow (stream of data)
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1,
    class_mode="categorical",
    shuffle=False)

# training
if(TRAIN):
    history = model.fit(
        training_generator,
        steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
        verbose=1,
        callbacks=[CSVLogger('log.csv', append=False, separator=",")]
    )

    model.save('modelBASIC.h5')

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='--')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))  # Epochs go every 2
    plt.savefig('accuracy_plot_BASIC.png')  # Save the accuracy plot
    plt.close()

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
    plt.title('Model Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))  # Epochs go every 2
    plt.savefig('loss_plot_BASIC.png')  # Save the loss plot
    plt.close()
