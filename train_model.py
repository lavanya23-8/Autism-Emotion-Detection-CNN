import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Image generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    'Dataset/train',
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical'
)

# Print class labels order
print("Class Order:", train_generator.class_indices)

# Load testing dataset
test_generator = test_datagen.flow_from_directory(
    'Dataset/test',
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical'
)

# CNN Model
model = Sequential()

# First CNN layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second CNN layer
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

# Output layer (6 emotions)
model.add(Dense(6,activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator
)

# Save trained model
model.save("emotion_model.h5")

print("Model saved successfully!")