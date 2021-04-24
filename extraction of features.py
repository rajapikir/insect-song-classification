import os
import csv
import librosa
import numpy as np
import pandas as pd
import contextlib
import wave
from matplotlib import pyplot as plt

#ekstraksi file wav menjadi fitur
mfccs=list()
lfccs=list()
mfcc_lfccs=list()
label=list()

#merubah file wav menjadi feature mfcc dan lfcc
for filenames in os.listdir(f"/content/drive/MyDrive/wav"):
  songname=f"/content/drive/MyDrive/wav/{filenames}"
  y,sr=librosa.load(songname,mono=True,duration=6.0)
  spec = np.abs(librosa.stft(y,n_fft=441,hop_length=220))
  specto = librosa.amplitude_to_db(spec, ref=np.max)
  melspec = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=441,hop_length=220
                                           )
  mfcc=librosa.feature.mfcc(S=librosa.power_to_db(melspec),sr=sr,n_mfcc=13)
  lfcc=librosa.feature.mfcc(S=specto,sr=sr,n_mfcc=13)
  #membuat label
  if "jeliteng" in filenames:
    label.append(1)
  elif "jerabang" in filenames:
    label.append(1)
  else :
    label.append(-1)
  #mengabungkan feature
  mfcc_lfcc=np.concatenate((lfcc,mfcc),axis=0)
  #merubah menjadi satu dimensi 
  mfcc_=mfcc.flatten()
  lfcc_=lfcc.flatten()
  mfcc_lfcc_=mfcc_lfcc.flatten()
  #menyimpan features ke list
  mfccs.append(mfcc_)
  lfccs.append(lfcc_)
  mfcc_lfccs.append(mfcc_lfcc_)
  
#memsihkan antara data latih dan tes 
from sklearn.model_selection import train_test_split
train_mfcc,test_mfcc,TrainLabel_mfcc,TestLabel_mfcc = train_test_split(mfccs,label,test_size=0.33,random_state=42)
train_lfcc,test_lfcc,TrainLabel_lfcc,TestLabel_lfcc = train_test_split(lfccs,label,test_size=0.33,random_state=42)
train_mfccl,test_mfccl,TrainLabel_mfccl,TestLabel_mfccl = train_test_split(mfcc_lfccs,label,test_size=0.33,random_state=42)

#memisahkan data train mfcc jangkrik dan non-jangkrik
TrainmfccJangkrik = []
TrainmfccnonJangkrik = []
k = 0
for r in TrainLabel_mfcc:
    if r==1:
      TrainmfccJangkrik.append(train_mfcc[k])
      k=k+1
    else:
      TrainmfccnonJangkrik.append(train_mfcc[k])
      k=k+1
#memisahkan data train lfcc jangkrik dan non-jangkrik
TrainlfccJangkrik = []
TrainlfccnonJangkrik = []
l = 0
for r in TrainLabel_lfcc:
    if r==1:
      TrainlfccJangkrik.append(train_lfcc[l])
      l=l+1
    else:
      TrainlfccnonJangkrik.append(train_lfcc[l])
      l=l+1
#memisahkan data train mfcc-lfcc jangkrik dan non-jangkrik
TrainmfcclJangkrik = []
TrainmfcclnonJangkrik = []
m = 0
for r in TrainLabel_mfccl:
    if r==1:
      TrainmfcclJangkrik.append(train_mfccl[m])
      m=m+1
    else:
      TrainmfcclnonJangkrik.append(train_mfccl[m])
      m=m+1
      
#merubah data list ke array 
TrainmfccJangkrik=np.array(TrainmfccJangkrik)
test_mfcc=np.array(test_mfcc)
TrainLabel_mfcc=np.array(TrainLabel_mfcc)
TestLabel_mfcc=np.array(TestLabel_mfcc)

TrainlfccJangkrik=np.array(TrainlfccJangkrik)
test_lfcc=np.array(test_lfcc)
TrainLabel_lfcc=np.array(TrainLabel_lfcc)
TestLabel_lfcc=np.array(TestLabel_lfcc)

TrainmfcclJangkrik=np.array(TrainmfcclJangkrik)
test_mfccl=np.array(test_mfccl)
TrainLabel_mfccl=np.array(TrainLabel_mfccl)
TestLabel_mfccl=np.array(TestLabel_mfccl)
