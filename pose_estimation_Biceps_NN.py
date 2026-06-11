import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras import layers, models # type: ignore
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Rescaling, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense,GlobalAveragePooling2D,AvgPool2D # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

# Parameters
image_size = (256, 256)
data_dir = r'C:\Users\archi\OneDrive\Desktop\Computer Vision AI Trainer\AI-Trainer\Videos\Biceps'
class_names = ["Top", "Bottom", "Middle"]
label_map = {name: idx for idx, name in enumerate(class_names)}

# 1. Load images and labels manually
X = []
y = []

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(img_path).resize(image_size)
            img_array = np.array(img) / 255.0  # normalize
            if img_array.shape == (256, 256, 3):  # filter out corrupt or grayscale
                X.append(img_array)
                y.append(label_map[class_name])
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

X = np.array(X)
y = to_categorical(y, num_classes=3)

# 2. Split into train/test using sklearn
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Define a simple CNN
def plot_results(history):
  fig, ax = plt.subplots(ncols=2, figsize=(12,4))
  ax[0].plot(history.history['accuracy'],label = 'train')
  ax[0].plot(history.history['val_accuracy'],label = 'test')
  ax[0].set_title('Accuracy')
  ax[0].legend(loc='lower right')
  ax[1].plot(history.history['loss'],label = 'train')
  ax[1].plot(history.history['val_loss'],label = 'test')
  ax[1].set_title('Loss')
  ax[1].legend(loc='upper right')
  plt.show()

def print_best_metrics(h):
  labels = ['training accuracy', 'validation accuracy', 'training loss', 'validation loss']
  keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
  prefix = ['highest','lowest']
  for i in range(4):
    metric = np.array(h[keys[i]])
    am = np.argmax((1.5-i)*metric)
    print(f'The {prefix[i//2]} {labels[i]} was {metric[am]:.4f} obtained in epoch {am+1}')
    print(f'The final {labels[i]} was {metric[-1]:.4f}')

def cnn_model(input_shape=(180, 180, 3), conv_units=(32, 64, 128), dense_units=100, dropout_rate=0.25, num_classes=3):
    model = tf.keras.models.Sequential()
    
    # Input and rescaling
    model.add(InputLayer(input_shape=input_shape))
    # model.add(Rescaling(1.0 / 255))q
    
    # Convolutional blocks
    for c in conv_units:
        model.add(Conv2D(c, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(c, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(GlobalAveragePooling2D())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


# 4. Compile and train
model = cnn_model(input_shape=(256, 256, 3), conv_units=(32, 64, 128), dense_units=128, dropout_rate=0.2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-3),
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(
    "best_cnn_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=48,
    callbacks=callbacks,
)

print_best_metrics(history.history)
plot_results(history)
