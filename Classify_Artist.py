from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import keras
import cv2
import tensorflow as tf
import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping


print(keras.__version__)


path = Path("test_data")
df = pd.read_csv(path/'artists.csv')
df.head()

print('Number of Artists:', len(df))


#check distribution of paintings
artists_df = df[['name', 'paintings']].groupby(['name'], as_index = False).sum()
names = artists_df.sort_values('paintings', ascending = False)[:50]


#remove spaces from names
images_dir = Path(path/'images/images')
artists = names['name'].str.replace(' ', '_').values

class CustomGenerator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([
            (cv2.imread(str(file_name)))
            for file_name in batch_x]), np.array(batch_y)

painting_list = []
for artist in artists:
    folder = Path(images_dir/artist)
    for subdir, dirs, files in os.walk(images_dir):
        for file in files:
            img = os.path.join(subdir, file)
            painting_list.extend([img])


#only works on windows
artist_list = []

for painting in painting_list:
    artist = painting.split('\\')[3]
    artist_list.extend([artist])

y = np.array(artist_list)  
X = np.array(painting_list)  

print(len(X))
print(len(y))


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(y)
print(y)

print(len(X))
print(len(y))


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=1)



batch_size = 32
training_batch_generator = CustomGenerator(X_train, y_train, batch_size)
validation_batch_generator = CustomGenerator(X_val, y_val, batch_size)
print(training_batch_generator)
print(validation_batch_generator)
training_size = len(X_train)
validation_size = len(X_val)
test_size = len(X_test)

print(training_size)
print(validation_size)
print(test_size)

# needed for first run
"""files = os.listdir(Path(images_dir/'Albrecht_Dürer'))
print(files)
for src in files:
    dst = src.replace('DuΓòá├¬rer', 'Dürer')
    os.rename(Path(images_dir/'Albrecht_Dürer'/src), Path(images_dir/'Albrecht_Dürer'/dst))"""

# Implement and Train A model based on VGG-16's architecture


# Generate the model
model = Sequential()
# Layer 1: Convolutional
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
# Layer 2: Convolutional
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 3: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 4: Convolutional
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 5: Convolutional
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 6: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 7: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 8: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 9: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 10: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 11: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 12: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 13: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 14: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 15: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 16: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 17: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 18: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 19: Flatten
model.add(Flatten())
# Layer 20: Fully Connected Layer
model.add(Dense(units=4096, activation='relu'))
# Layer 21: Fully Connected Layer
model.add(Dense(units=4096, activation='relu'))
# Layer 22: Softmax Layer
model.add(Dense(units=8, activation='softmax'))
print(model.summary())



model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit_generator(generator=training_batch_generator,
                    steps_per_epoch=int(training_size // batch_size),
                    epochs=6,
                    verbose=1,
                    validation_data=validation_batch_generator,
                    validation_steps=int(validation_size // batch_size)
                   )

model.save(Path('model.h5'))
loss_train = model.history.history['loss']
loss_val = model.history.history['val_loss']

