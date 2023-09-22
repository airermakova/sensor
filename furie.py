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

## DEFINE AVAILABLE ENTRANCE PARAMETERS
taumin = 40e-6  # Minimum time constant
taumax = 300e-6  # Maximum time constant
fmin = 8e3  # Minimum frequency (can be set from 8e3 to 48e3)
fmax = fmin + 4e3  # Maximum frequency
lock = threading.Lock()
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
dft = []
signal = []
t_f = []
koef = []
t = []
cnt = 0


def get_magnitude(signal_f):
    magnitude = []
    for s in signal_f:
        dft1_real = numpy.real(s)
        dft1_imag = numpy.imag(s)
        m = numpy.sqrt((dft1_real * dft1_real) + (dft1_imag * dft1_imag))
        magnitude.append(m)
    return magnitude


def get_transform_frequencies(timerange, values):
    # Discrete Fourier Transform sample frequencies for manual plot
    return fftfreq(values, timerange / values)[: values // 2]


def show_plot(t_f, signal_f, magnitude, cnt, tau, f0):
    pyplot.plot(t_f, (signal_f[:cnt]), "-b")
    pyplot.legend([f"fft transformation. tau:{tau} f0:{f0}."])
    pyplot.grid()
    pyplot.show()
    pyplot.plot(t_f, (magnitude[:cnt]), "-b")
    pyplot.legend([f"magnitude spectrum. tau:{tau} f0:{f0}."])
    pyplot.grid()
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
    # signal = numpy.cos(2 * numpy.pi * f0 * t)
    signal_f = fft(signal)
    t_f = get_transform_frequencies(timerange, values)
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


# PREPARE MANUAL DFT TRANSFORM
def dft_transform(timerange, values, tau, f):
    global signal
    global t_f
    global dft
    global koef
    global t
    global cnt
    prev = psutil.cpu_percent()
    t = numpy.linspace(0, timerange, values)
    k = numpy.linspace(0, values - 1, values)
    print(k)
    signal = numpy.multiply(numpy.exp(-t / tau), numpy.cos(2 * numpy.pi * f * t))
    t_f = get_transform_frequencies(timerange, values)
    koef = numpy.multiply(((1j * 2 * numpy.pi) / values), k)
    manual_fourie = numpy.vectorize(manual_ft_2)
    dft = manual_fourie(k)
    magnitude = get_magnitude(dft)
    show_plot(t_f, dft, magnitude, int(values // 2), tau, f)


def manual_ft_1(s, t):
    global koef
    global cnt
    out = numpy.multiply(s, numpy.exp(numpy.multiply(t, cnt)))
    retval = numpy.sum(out)
    return retval


def manual_ft_2(k):
    global signal
    global koef
    global cnt
    global dft
    manual_fourie = numpy.vectorize(manual_ft_1)
    retval = numpy.sum(manual_fourie(signal, koef))
    cnt = cnt + 1
    print(cnt)
    return retval


# END OF THE BLOCK TO GENERATE INPUT SIGNAL


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
                print(
                    "entrance arguments: -t tau, -f frequency -p period. In case -p is not provided period calculated as 1/f"
                )
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
    # auto_fft_transform(1, 6000, tau, f, 1)
    # run_manual_dft(1, 6000, tau, f)
    dft_transform(1, 15000, tau, f)


if __name__ == "__main__":
    main(sys.argv[1:])
