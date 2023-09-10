import getopt
import math
import os
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
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal.windows import blackman
import psutil
import scipy

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
ffttaumax = taumax
fs = 1100


def get_magnitude(signal_f):
    magnitude = []
    for s in signal_f:
        dft1_real = numpy.real(s)
        dft1_imag = numpy.imag(s)
        m = numpy.sqrt((dft1_real * dft1_real) + (dft1_imag * dft1_imag))
        magnitude.append(m)
    return magnitude


def get_transform_frequencies(signal, timerange, values):
    # Discrete Fourier Transform sample frequencies for manual plot
    return fftfreq(timerange, signal)[: values // 2]


def show_plot(t_f, signal_f, magnitude, cnt, tau, f0):
    pyplot.plot(t_f, (signal_f[:cnt]), "-b")
    pyplot.legend([f"fft transformation. tau:{tau} f0:{f0}."])
    pyplot.grid()
    pyplot.show()
    pyplot.plot(t_f, (magnitude[:cnt]), "-b")
    pyplot.legend([f"magnitude spectrum. tau:{tau} f0:{f0}."])
    pyplot.grid()
    pyplot.show()
    # plotting the magnitude spectrum of the signal
    pyplot.magnitude_spectrum(signal_f, color="green")
    pyplot.title("Magnitude Spectrum of the Signal")
    pyplot.show()


## PREPARE SIGNAL SIMULATED DATA
def auto_fft_transform(timerange, values, tau, f0, showplot=0):
    global label
    global data
    global signals

    prev = psutil.cpu_percent()
    # Creating vectors of time and values
    t = numpy.linspace(0, timerange, values)
    signal = numpy.multiply(numpy.exp(-t / tau), numpy.cos(2 * numpy.pi * f0 * t))
    # Get DFT with python scipy library
    signal_f = fft(signal)
    t_f = get_transform_frequencies(signal_f, timerange, values)
    magnitude = get_magnitude(signal_f)
    mem = sys.getsizeof(magnitude)
    act = psutil.cpu_percent()
    increase = 0
    if act > prev:
        increase = act - prev
    print(
        f"Scipy DFT and magintude extraction get: virtual memory used to store fft array: {mem} bytes, cpu load increase:{increase}"
    )
    # Get DFT manually
    if showplot == 1:
        pyplot.plot(t, signal, "-r")
        pyplot.legend([f"real graph tau:{tau} f0:{f0}"])
        pyplot.grid()
        pyplot.show()
        show_plot(t_f, signal_f, magnitude, (values // 2), tau, f0)


## PREPARE SIGNAL SIMULATED DATA
def manual_fft_transform(timerange, values, tau, T, showplot=0):
    global label
    global data
    global signals

    signal = []
    prev = psutil.cpu_percent()

    t = numpy.linspace(0, timerange, values)
    for i in t:
        signal.append(
            B
            * math.pow((math.e), (-i * T / ffttaumax))
            * math.cos(2 * math.pi * fss * i * T)
        )
    magnitude = get_magnitude(signal)
    # Get DFT with python scipy library
    act = psutil.cpu_percent()
    mem = sys.getsizeof(signal)
    increase = 0
    if act > prev:
        increase = act - prev
    print(
        f"Manual DFT and magintude extraction get: virtual memory used to store fft array: {mem}bytes, cpu load increase:{increase}"
    )

    # Get DFT manually
    if showplot == 1:
        show_plot(t, signal, magnitude, (values), tau, f0)


# END OF THE BLOCK TO GENERATE INPUT SIGNAL


# function to get Furie sequence for one measurement interval for fs and tau
def createonesequence(num):
    global fs
    global ths


# END OF FFT SECTION


def main(argv):
    global fs
    global ths

    argv = sys.argv[1:]
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    print(args)
    tau = 0.0
    f = 0.0
    T = 0
    for i in range(0, len(opts)):
        try:
            if opts[i] == "-h":
                print("test.py -i <inputfile> -o <outputfile>")
                sys.exit()
            elif opts[i] == "-t":
                tau = float(args[i])
            elif opts[i] in ("-f", "--f0"):
                f = float(args[i])
                T = 1 / f
            elif opts[i] in ("-p", "--period"):
                T = float(args[i])
        except Exception:
            continue
    auto_fft_transform(1, 300, tau, f, 1)
    manual_fft_transform(1, 300, tau, T, 1)

    # for i in range(0, K):
    #    thread = threading.Thread(target=createonesequence, args=(i,))
    #    thread.start()
    # while len(ths) > 1:
    #    time.sleep(1)
    # for i in finalsequence:
    #    pyplot.plot(i)
    # pyplot.show()
    # print(finalsequence)


if __name__ == "__main__":
    main(sys.argv[1:])
