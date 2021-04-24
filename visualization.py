# untuk membuat nama csv file dan merubah data menjadi vektor mfcc dan lfcc
for filenames in os.listdir(f"./wav/"):
    filename = filenames.replace(".wav","")
    filename = filename+".csv"
    #  merubah data menjadi vektor 
    songname = f"./wav/{filenames}"
    #ekstraksi fitur
    y, sr = librosa.load(songname, mono=True, duration = 5.0 )
    spec = np.abs(librosa.stft(y,n_fft=441,hop_length=220))
    specto = librosa.amplitude_to_db(spec, ref=np.max)
    melspec = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=441,hop_length=220
                                           )
    mfcc=librosa.feature.mfcc(S=librosa.power_to_db(melspec),sr=sr,n_mfcc=13)
    lfcc=librosa.feature.mfcc(S=specto,sr=sr,n_mfcc=13)
    mfcc_lfcc=np.concatenate((lfcc,mfcc),axis=0)
    #tranpose fitur 
    mfcc = mfcc.transpose()
    lfcc = lfcc.transpose()
    mfcc_lfccs = mfcc_lfcc.transpose()
    #erubah fitur menjadi list 
    features_mfcc = mfcc.tolist()
    features_lfcc = lfcc.tolist()
    features_mfccl = mfcc_lfccs.tolist()
    # menyimpan feature mfcc ke dalam file csv
    with open("mfcc_"+filename, 'w', newline = '') as file:
      writer = csv.writer(file)
      writer.writerows(features_mfcc)
    # menyimpan feature lfcc ke dalam file csv
    with open("lfcc_"+filename, 'w', newline = '') as file:
      writer = csv.writer(file)
      writer.writerows(features_lfcc)
    # menyimpan feature mfcc-lfcc ke dalam file csv
    with open("mfcclf_"+filename, 'w', newline = '') as file:
      writer = csv.writer(file)
      writer.writerows(features_mfccl)
      
# membuat header feature mfcc pada csv
header_mfcc = []
header_lfcc = []
header_mfcc_lfcc = []
for i in range(1, 14):
    heade = "mfcc " + str(i)
    header_mfcc.append(heade)
for i in range(1, 14):
    heade = "lfcc " + str(i)
    header_lfcc.append(heade)
for i in range(1, 27):
    heade = "mfcc_lfcc" + str(i)
    header_mfcc_lfcc.append(heade)
for filen in os.listdir(f"./file_mfcc/"):
    hui = pd.read_csv(f"./file_mfcc/{filen}", header= None )
    hui.to_csv(f"./file_mfcc/{filen}" , header = header_mfcc, index=False)
for filen in os.listdir(f"./file_lfcc/"):
    hui = pd.read_csv(f"./file_lfcc/{filen}", header= None )
    hui.to_csv(f"./file_lfcc/{filen}" , header = header_lfcc, index=False)
for filen in os.listdir(f"./file_mfcc_lfcc/"):
    hui = pd.read_csv(f"./file_mfcc_lfcc/{filen}", header= None )
    hui.to_csv(f"./file_mfcc_lfcc/{filen}" , header = header_mfcc_lfcc, index=False)
    
# membuat grafik visualisasi mfcc untuk seluruh dataset
n = len(filen)
for filen in os.listdir(f"./file_mfcc/"):
    df = pd.read_csv(f"./file_mfcc/{filen}")
    plt.figure()
    plt.ylabel('nilai')
    plt.xlabel('waktu')
    plt.title(filen)
    for s in header_mfcc:
            plt.plot(df[s], label = s)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
# membuat grafik visualisasi lfcc untuk seluruh dataset
n = len(filen)
for filen in os.listdir(f"./file_lfcc/"):
    df = pd.read_csv(f"./file_lfcc/{filen}")
    plt.figure()
    plt.title(filen)
    for s in header_lfcc:
            plt.plot(df[s], label = s)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #plt.savefig('myfilename%03d.png'%(n))
    
# membuat grafik visualisasi mfcc-lfcc untuk seluruh dataset
n = len(filen)
for filen in os.listdir(f"./file_mfcc_lfcc/"):
    df = pd.read_csv(f"./file_mfcc_lfcc/{filen}")
    plt.figure()
    plt.title(filen)
    for s in header_mfcc_lfcc:
            plt.plot(df[s], label = s)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #plt.savefig('myfilename%03d.png'%(n))
    
import matplotlib.pyplot as plot
from scipy.io import wavfile
import scipy.io

 

# membuat visualisasi spectogram untuk setiap dataset
for polen in os.listdir(f"./wav/"):
    wavfile = f"./wav/{polen}"
    samplingFrequency, signalData = scipy.io.wavfile.read(wavfile)
    # Plot the signal read from wav file
    plot.figure(figsize=(9,9))
    plot.subplot(212)
    plot.title('Spectrogram '+ str(polen))
    plot.specgram(signalData,Fs = samplingFrequency)    
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.ylim([0, 22000])
