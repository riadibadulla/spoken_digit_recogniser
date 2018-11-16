try:
    # Python2
    from Tkinter import *
except ImportError:
    # Python3
    from tkinter import *

root = Tk()
root.geometry('800x500')
root.title("Speech Recognition Systems - LSTM")
root.resizable(0,0)
import Testing
import spectrogram
T = Text(root, height=3, width=100, highlightbackground = 'black')
T.place(relx=0.02,rely=0.8)


waveLabel = Label(root, text = "Salam Dünya", borderwidth = 2, relief ="groove")
waveImage = PhotoImage()
specLabel = Label(root, text = "Salam Dünya", borderwidth = 2, relief ="groove")
specImage = PhotoImage()
#T.pack()


def record():
    global waveImage, specImage
    Testing.record()
    T.delete(1.0,END)
    result = "Recognised Digit:" + str(Testing.recognisedNumber) + "\nWith Probability of:" + \
             str(Testing.probabilityOfRecognisedDegit*100) + "%"
    T.insert(END, result)
    spectrogram.createPlots()
    print("saved")

    waveImage = PhotoImage(file='wave.ppm', height=250, width=390)
    waveLabel.configure(image=waveImage)
    waveLabel.place(relx=0.01, rely=0.01)
    specImage = PhotoImage(file='spec.ppm', height=250, width=360)
    specLabel.configure(image=specImage)
    specLabel.place(relx=0.53, rely=0.01)

def train():
    trainRoot = Tk()
    trainRoot.geometry('500x600')
    trainRoot.title("Training stage")
    trainRoot.resizable(0, 0)

    trainingCMDText = Text(trainRoot,height=23, width=68, highlightbackground = 'black')
    trainingCMDText.place(relx=0.01, rely=0.4)

    trainingItersLabel = Label(trainRoot, text = "Training Iters:")
    trainingItersLabel.place(relx=0.01, rely=0.05)
    trainingIters = Entry(trainRoot, highlightbackground = 'blue')
    trainingIters.place(relx=0.23, rely=0.05)

    LearningRateLabel = Label(trainRoot, text="Learning Rate:")
    LearningRateLabel.place(relx=0.01, rely=0.1)
    LearningRate = Entry(trainRoot, highlightbackground='blue')
    LearningRate.place(relx=0.23, rely=0.1)
    trainRoot.mainloop()
    #import Training
    print("Training")

b = Button(root, text="Record", command=record, width = 10, height = 2)
b.place(relx=0.08,rely=0.65)

root.mainloop()


