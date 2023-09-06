import getopt
import math
import sys
import threading
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTMCell, Masking
from keras.layers.rnn import ConvLSTM1D
import numpy
from matplotlib import pyplot
import time
from keras.layers import LSTM
import json
from sklearn.model_selection import train_test_split
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman


# BLOCK TO GENERATE INPUT SIGNAL
M = 100000  # Number of sequences
taumin = 40e-6  # Minimum time constant
taumax = 300e-6  # Maximum time constant
fmin = 8e3  # Minimum frequency (can be set from 8e3 to 48e3)
fmax = fmin + 4e3  # Maximum frequency
samples_per_period = 100  # Nominal number of samples per period (editable)
fc = samples_per_period * fmax  # Sampling frequency
W = 5 * taumin  # Observation window
## COMPUTE THE TIME CONSTANTS AND FREQUENCIES
tau = taumin + (taumax - taumin)
f0 = fmin + (fmax - fmin)

## CREATE THE TIME ARRAY
N = numpy.floor(W * fc)
t = 1 / fc * numpy.arange(0, N - 1, 1)

## BUILD THE DATASET
label = []
data = []
signals = []


## PREPARE SIGNAL SIMULATED DATA
def prepare_signal_simulation(timerange, values, koeff, showplot=0):
    global label
    global data
    global signals

    tau = taumin + (taumax - taumin) * koeff
    f0 = fmin + (fmax - fmin) * koeff
    # Creating vectors of time and values
    t = numpy.linspace(0, timerange, values)
    signal = numpy.multiply(numpy.exp(-t / tau), numpy.cos(2 * numpy.pi * f0 * t))
    signal = numpy.multiply(2, numpy.cos(2 * numpy.pi * f0 * t))

    # Get DFT with python fft library
    signal_f = fft(signal)
    t_f = fftfreq(values, timerange)[: values // 2]

    if showplot == 1:
        pyplot.plot(t, signal, "-r")
        pyplot.legend([f"real graph"])
        pyplot.grid()
        pyplot.show()
        pyplot.plot(t_f, (signal_f[: values // 2]), "-b")
        pyplot.legend(["fft"])
        pyplot.grid()
        pyplot.show()


# END OF THE BLOCK TO GENERATE INPUT SIGNAL

# FFT SECTION

## DEFINE AVAILABLE ENTRANCE PARAMETERS
lock = threading.Lock()
ths = []
N = 100  # number of values in sequence
fex = 1000  # excitation frequency (Hz)
K = 10  # the number of acquisitions
fb = 1000  # local oscillator frequency (Hz)
Tburst = 10  # excitation burst duration (ms)
fss = 100  # mixed signal frequency (Hz)
ffttaumin = Tburst / 2  # minimum tau
Vex = 10  # excitation voltage (Volts)
Rq = 1  # piezoelectric material resistance (Ohm)
Cq = 1  # piezoelectric material capacitance (Farad)
Lq = 1  # piezoelectric material inductance ()
B = Vex / Rq

fs = (
    []
)  # measured quartz resonant frequency (Hz) (measurement occurs every K acquisitions)
ffttaumax = []  # measured tau (measurement occurs every K acquisitions)

finalsequence = []
checksequence = []


# service function to load fs and taus from input file array
def getResonantFrequencyAndTau():
    global fs
    global ffttaumax
    f = open("tauandfs.json")
    data = json.load(f)
    fs = data["fs"]
    ffttaumax = data["tau"]
    print(fs)
    print(ffttaumax)


# function to prepare outcome matrix
def prepareoutputmatrix():
    global finalsequence
    global checksequence
    f = []
    for i in range(0, N):
        f.append(0)
    for i in range(0, len(fs)):
        finalsequence.append(f)
        checksequence.append(f)


# function to get Furie sequence for one measurement interval for fs and tau
def createonesequence(num):
    global fs
    global finalsequence
    global checksequence
    global ths
    try:
        lock.acquire()
        ths.append(1)
        lock.release()
        sequence = []
        yf = []
        fss = fs[num] - fb
        Ts = 1 / fs[num]
        for i in range(0, N):
            y = B * math.pow((math.e), (i / ffttaumax[num])) * math.cos(2 * math.pi * i)
            I = (
                B
                * math.pow((math.e), (-i * Ts / ffttaumax[num]))
                * math.cos(2 * math.pi * fss * i * Ts)
            )
            sequence.append(I)
            yf.append(y)
        lock.acquire()
        checksequence.insert(num, yf)
        finalsequence.insert(num, sequence)
        ths.pop()
        lock.release()
    except Exception as e:
        print(f"Exception in thread {num}: {e}")


# END OF FFT SECTION


def main(argv):
    global finalsequence
    global fs
    global ths
    prepare_signal_simulation(3, 300, 1000, 1)
    # getResonantFrequencyAndTau()
    # prepareoutputmatrix()
    for i in range(0, K):
        thread = threading.Thread(target=createonesequence, args=(i,))
        # thread.start()
    # while len(ths) > 1:
    #    time.sleep(1)
    # for i in finalsequence:
    #    pyplot.plot(i)
    # pyplot.show()
    # print(finalsequence)


if __name__ == "__main__":
    main(sys.argv[1:])
