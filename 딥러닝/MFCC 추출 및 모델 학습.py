import librosa
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def extract_features(file_path, mfcc_max_padding):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = mfcc_max_padding - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("실패")
        return None
    return mfccs


dataset_dir = "데이터셋 파일 경로"
files = os.listdir(dataset_dir)
dataset = []
percentage = 0

max_padding = 800
for file in files:
    label = file.split("_")[0]
    mfcc = extract_features(os.path.join(dataset_dir, file), mfcc_max_padding=max_padding)
    if mfcc is not None:
        percentage += (1/len(files))*100
        os.system('cls')
        print(round(percentage),'%')
        dataset.append((mfcc, label))

# 데이터셋을 특징과 레이블로 분리 (x가 자료, y가 정답)
x = np.array([data[0] for data in dataset])
y = np.array([data[1] for data in dataset])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


os.system('cls')
input_shape = (x_train.shape[1], x_train.shape[2])
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2, padding='same'),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2, padding='same'),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2, padding='same'),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_train_array = np.array(y_train_encoded)
y_test_array = np.array(y_test_encoded)

Training = model.fit(x_train, y_train_array, epochs=100, batch_size=32, validation_data=(x_test, y_test_array), callbacks=callbacks)
plt.plot(Training.history['accuracy'], label='Train Accuracy')
plt.plot(Training.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("final_model.keras")
