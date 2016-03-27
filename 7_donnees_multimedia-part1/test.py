import scipy.io.wavfile 
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt

# load audio file wav, normalise it and visualize it
rate,data = scipy.io.wavfile.read("es01.wav")
N = data.size
fe = rate # frequency sampling
#N=1000
#fe = 10.0
Te = 1.0/fe
t = Te*np.arange(N)
f = fe*np.arange(N)/N
#data = 0.7*np.cos(2*np.pi*0.4*t)
data_centered = data-np.mean(data)
data_norm = data_centered/np.amax(np.abs(data_centered))
fft = scipy.fftpack.fft(data)
fft_norm = scipy.fftpack.fft(data_norm)
I = f<10000
fft_pruned = fft.copy()
fft_pruned[I] = 0
data_reconstruct = np.real(scipy.fftpack.ifft(fft))
fig,ax = plt.subplots(3,2)
ax[0,0].plot(t,data)
ax[0,0].set_title("signal")
ax[0,0].set_xlabel('s')
ax[1,0].plot(t,data_norm)
ax[1,0].set_title("signal normalized")
ax[1,0].set_xlabel('s')
ax[2,0].plot(t,data_reconstruct)
ax[2,0].set_title("signal reconstruct")
ax[2,0].set_xlabel("s")
ax[0,1].plot(f,np.abs(fft))
ax[0,1].set_title("fourier")
ax[0,1].set_xlabel('Hz')
ax[1,1].plot(f,np.abs(fft_norm))
ax[1,1].set_title("fourier normalized")
ax[1,1].set_xlabel('Hz')
ax[2,1].plot(f,np.abs(fft_pruned))
ax[2,1].set_title("fourier pruned")
ax[2,1].set_xlabel("Hz")
scipy.io.wavfile.write("es01_norm.wav",rate,data_norm)
scipy.io.wavfile.write("es01_reconstruct.wav",rate,data_reconstruct)


plt.show()

