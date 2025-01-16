from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
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
MODEL_SUMMARY_FILE = "model_summary_pro.txt"

# hyperparameters
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128
RGB = 3
EPOCHS = 25
BATCH_SIZE = 4

# create CNN model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, RGB), activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='tanh'))
model.add(Conv2D(64, (3, 3), padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])

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

    model.save('modelPRO.h5')

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
    plt.savefig('accuracy_plot_PRO.png')  # Save the accuracy plot
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
    plt.savefig('loss_plot_PRO.png')  # Save the loss plot
    plt.close()
