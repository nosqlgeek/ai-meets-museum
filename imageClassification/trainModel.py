from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Create a pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



# Add a new output layer
# Classes count
num_classes = 3

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create a new model with the new output layer
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Create an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


train = pd.read_csv('./imageClassification/dataset/train.csv')
test = pd.read_csv('./imageClassification/dataset/test.csv')

# Flow the training and testing data through the ImageDataGenerators
train_generator = train_datagen.flow_from_dataframe(dataframe=train, directory='./imageClassification/images/', x_col='filename', y_col='label', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(dataframe=test, directory='./imageClassification/images/', x_col='filename', y_col='label', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_generator)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')


# Output:
#Epoch 1/10
#1/1 [==============================] - 8s 8s/step - loss: 1.0310 - accuracy: 0.6000 - val_loss: 5.5994 - val_accuracy: 0.2500
#Epoch 2/10
#1/1 [==============================] - 1s 814ms/step - loss: 7.3591 - accuracy: 0.8000 - val_loss: 49.9760 - val_accuracy: 0.5000
#
# loss: so gering wie möglich => model hat richtiges label gefunden
# val_loss: validation loss is increasing dramatically, which indicates that the model is not generalizing well to new data.
# val_accuracy: validation accuracy remains constant at 0.5, which is the same as random guessing for a binary classification problem.
#