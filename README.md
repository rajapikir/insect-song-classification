# insect-song-classification
mengidentifikasi suara suara  jangkrik dengan metode novelty detection mengunakan svm one class
file terdiri dari suara jangkring, tonggeret dan anjing tanah.
svm one class berbeda dengan svm biasa dikarenakan hanya membutuhkan satu kelas untuk membuat data latihnya dan membuat densitas dari data latih. data latih 
terdiri dari suara jangkrik, sedangkan data tes terdiri dari ketiga jenis suara seranga-seranga tersebut. ekstraksi fitur mengunakan mfcc dan lfcc. kedua fitur tersebut 
berdasarkan spectral dari sinyal suara yang dihasilkan. membedakan kedua jenis fitur tersebut adalah filter yang digunakan mfcc mengunakan melspectogram dan lfcc linier. melspectogram disini mengikuti persepsi telingga manusia. 

hasil yang didapatkan setelah melalukan model adalah


![image](https://user-images.githubusercontent.com/83129067/115964993-46947300-a551-11eb-97e1-e6e9a0ea1785.png)


