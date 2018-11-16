from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from PIL import Image


# read audio samples
def createPlots():
    input_data = read("0_new.wav")
    audio = input_data[1]
    numberOfSamples = len(audio)
    # plot the first 1024 samples
    plt.figure(figsize=(3.9,2.5))
    plt.plot(audio[0:numberOfSamples])
    # label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    # set the title
    plt.title("Sample Wav")
    # display the plot
    #plt.show()
    plt.savefig('wave.png')
    plt.close()
    plt.figure(figsize=(3.6, 2.4))
    plt.specgram(audio, NFFT=256, Fs=2, Fc=0, noverlap=128,
                 cmap=None, xextent=None, pad_to=None, sides='default',
                 scale_by_freq=None, mode='default', scale='default')
    #plt.show()
    plt.savefig('spec.png')

    im = Image.open("wave.png")
    im.save("wave.ppm")
    im = Image.open("spec.png")
    im.save("spec.ppm")
