import scipy.io.wavfile 
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab

# load audio file wav
audio_name = "hauteur.wav"
print "* audio file : ",audio_name
rate,data = scipy.io.wavfile.read(audio_name)
if len(data.shape)>1:
    data = data[:,0]
N = data.size
print "* number samples : ",N
fe = rate # frequency sampling
print "* sampling frequency : ",fe
#N=1000
#fe = 5.0
Te = 1.0/fe
t = Te*np.arange(N) # t = time support [0:Te:N*Te]
f = fe*np.arange(N)/N-0.5*fe # frequency support [-fe/2:fe/2]
f2 = fe*0.5*np.arange(N)/N # frequency support [0:fe/2] (for dct)
#data = 0.7*np.cos(2*np.pi*0.005*t)
#data = 0.5*np.arange(N)

# perform fft and fftshift to get fft between [-fe/2:fe/2]
fft = scipy.fftpack.fftshift(scipy.fftpack.fft(data))

# perform dct
dct = scipy.fftpack.dct(np.float32(data))

# low pass filter
fc = 2000.0
I = (f<-fc) | (f>fc)
fft_filtered = fft.copy()
fft_filtered[I] = 0.0
I2 = f2>fc
dct_filtered = dct.copy()
dct_filtered[I2] = 0.0

# reconstruct low pass filtered
data_reconstruct_fft = np.real(scipy.fftpack.ifft(scipy.fftpack.ifftshift(fft_filtered)))
data_reconstruct_dct = np.real(scipy.fftpack.idct(dct_filtered))
scipy.io.wavfile.write("low_pass.wav",rate,data_reconstruct_fft)

# plot signals
fig,ax = plt.subplots(2,3)
ax[0,0].plot(t,data)
ax[0,0].set_title("signal")
ax[0,0].set_xlabel('s')
ax[1,0].plot(t,data_reconstruct_dct)
ax[1,0].set_title("signal reconstruct")
ax[1,0].set_xlabel("s")
ax[0,1].plot(f,np.abs(fft))
ax[0,1].set_title("dft")
ax[0,1].set_xlabel('Hz')
ax[1,1].plot(f,np.abs(fft_filtered))
ax[1,1].set_title("dft filtered")
ax[1,1].set_xlabel("Hz")
ax[0,2].plot(f2,np.abs(dct))
ax[0,2].set_title("dct")
ax[0,2].set_xlabel('Hz')
ax[1,2].plot(f2,np.abs(dct_filtered))
ax[1,2].set_title("dct filtered")
ax[1,2].set_xlabel("Hz")
plt.show()

# handcrafted specgram
nfft=256
novlp=128
hand_specgram = np.asarray([])
for i in np.arange(0,N-nfft,(nfft-novlp)):
    window = data[i:i+nfft]
    fft_window = scipy.fftpack.fft(window) # fft between [0:fe]
    fft_window = fft_window[0:int(fft_window.size/2)+1] # fft between [0:fe/2]
    fft_window = np.flipud(fft_window)
    fft_window = 20*np.log10(np.abs(fft_window)+0.01)
    fft_window = np.transpose(fft_window)[:,np.newaxis]
    if i==0:
        hand_specgram = fft_window 
    else:
        hand_specgram = np.hstack((hand_specgram,fft_window))
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.imshow(hand_specgram/np.amax(hand_specgram.flatten()))
ax1.set_title("hand crafted specgram")
ax2 = fig.add_subplot(2,1,2)
Pxx, freqs, bins, im = ax2.specgram(data,NFFT=nfft,Fs=fe,noverlap=novlp)
ax2.set_title("matplotlib specgram")
plt.show()


