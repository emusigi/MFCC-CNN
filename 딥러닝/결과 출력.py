import librosa
import sounddevice as sd
import soundfile as sf
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# 녹음 설정
duration = 10       # 녹음 시간 (초)
fs = 44100          # 샘플링 레이트


print("녹음을 시작")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True)

save_path = "저장할 위치"
if not os.path.exists(save_path):
    os.makedirs(save_path)

        # 파일명 입력받기
filename = "입력"+".wav"
file_path = os.path.join(save_path, filename)

sf.write(file_path, recording, fs)
print(f"녹음 파일이 '{file_path}'에 저장되었습니다.")

        # 녹음 데이터 로드
y, sr = librosa.load(file_path)

input_file_path = "저장한 파일 위치"  

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

mfcc = extract_features(input_file_path, mfcc_max_padding=800)

model = tf.keras.models.load_model('모델 keras 위치') 

mfcc_input = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
predictions = model.predict(mfcc_input)

predicted_label = np.argmax(predictions) #예측값 출력

#예측값 디코딩
print(str(predicted_label*10+30)+'개')
