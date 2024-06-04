import librosa
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def extract_features(file_path, mfcc_max_padding):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_best')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = mfcc_max_padding - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("실패:", file_path)
        return None 
    
    return mfccs

input_file_path = "C:/Users/emusigi/OneDrive/바탕 화면/R&E 파일/풀-데이타르/40_5_40_97.wav"  
mfcc = extract_features(input_file_path, mfcc_max_padding=800)

model = tf.keras.models.load_model('model.keras') 

mfcc_input = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
predictions = model.predict(mfcc_input)

predicted_label = np.argmax(predictions) #예측값 출력

#예측값 디코딩
print(str(predicted_label*10+20)+'개')