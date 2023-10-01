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

## ARRAYS TO GET FOURIER TRANSFORM
fs = 1100
dft = []
signal = []
t_f = []
koef = []
t = []
cnt = 0
# VALUES DEFINED
tc = 1 / 1e6
tau = 70e-6
ndft = 4000
f00 = 50e3
Vmax = 0.5
freq = [0] * 20
freq2 = [0] * 20
meanfreq = [0] * 10
meanfreq2 = [0] * 10
meanmeanfreq = [0] * 20
stdmeanfreq = [0] * 20
meanmeanfreq2 = [0] * 20
stdmeanfreq2 = [0] * 20
meanf = [0] * 20
stdfreq = [0] * 20
meanf2 = [0] * 20
stdfreq2 = [0] * 20
f0v = [0] * 20

n = numpy.minimum(math.floor(5 * tau / tc), ndft)
t = numpy.multiply(numpy.linspace(0, n - 1, n - 1), tc)
noise = [1e-3, 5e-3, 2e-2, 1e-1]

for kplot in range(1, 4):
    Vmax = 0.5
    for kk in range(1, 20):
        f0 = f00 + (kk - 10) * 250 / 20 + 3.333
        zeta = 1 / tau / 2 / numpy.pi / f0
        f0v[kk]=f0
        fas = (
            tc * f0 * 2 * numpy.pi
        )  # errore di fase dovuto alla determinazione del picco
        v = numpy.multiply(
            Vmax,
            numpy.exp(-t / tau),
            numpy.sin(2 * numpy.pi * f0 * t + fas * (numpy.random(1) - 0.5) * 0),
        )
        Spower = numpy.sum(numpy.power(v, 2)) / n
        for j in range(1, 20):
            for i in range(1, 20):
                vv = v + numpy.random.randn(1, n) * Vmax * noise[kplot]
                Npower = Vmax * noise[kplot]
                nn = ndft - n
                rn = nn + i
                vl = [vv, numpy.zeros(rn - 1)]
                v2 = [vv, numpy.zeros(nn - 1)]
                v1 = v1 + 1.5
                v2 = v2 + 1.5
                V = fft(vl - numpy.mean(vl))
                V2 = fft(v2 - numpy.mean(v2))
                vm = vl - numpy.mean(vl)
                vm2 = v2 - numpy.mean(v2)
                NN = len(vl)
                for fk in range(1, len(NN)):
                    for ti in range(1, len(NN)):
                        V[fk] = numpy.sum(
                            vm[ti] * numpy.cos(ti * fk * 2 * numpy.pi / NN)
                        ) + j * numpy.sum(
                            vm(ti) * numpy.sin(ti * fk * 2 * numpy.pi / NN)
                        )
                        V2[fk] = numpy.sum(
                            vm2[ti] * numpy.cos(ti * fk * 2 * numpy.pi / NN)
                        ) + j * numpy.sum(
                            vm2(ti) * numpy.sin(ti * fk * 2 * numpy.pi / NN)
                        )
                ini = 1
                a = numpy.max(
                    numpy.abs(numpy.power(V[ini : math.floor((n + rn) / 2)], 2))
                )
                b = numpy.max(
                    numpy.abs(numpy.power(V[ini : math.floor((n + rn) / 2)], 2))
                )
                a2 = numpy.max(
                    numpy.abs(numpy.power(V[ini : math.floor((n + nn / 2) / 2)], 2))
                )
                b2 = numpy.max(
                    numpy.abs(numpy.power(V2[ini : math.floor((n + nn / 2) / 2)], 2))
                )
                f = numpy.multiply(numpy.linspace(0, n + rn, n + rn), 1 / tc / (n + rn))
                f2 = numpy.multiply(
                    numpy.linspace(0, n + nn, n + nn), 1 / tc / (n + nn)
                )
                freq[i] = (
                    f(b + ini - 1)
                    / numpy.sqrt(1 - 2 * zeta ^ 2)
                    * numpy.sqrt(1 - zeta ^ 2)
                    - f0
                )
                freq2[i] = (
                    f2(b2 + ini - 1)
                    / numpy.sqrt(1 - 2 * zeta ^ 2)
                    * numpy.sqrt(1 - zeta ^ 2)
                    - f0
                )
                # cos
                freq[i] = f(b + ini - 1) * numpy.sqrt(1 - 0 * zeta ^ 2) - f0
                freq2[i] = f2(b2 + ini - 1) * numpy.sqrt(1 - 0 * zeta ^ 2) - f0
            SNR = 10 * numpy.log10(Spower / Npower)
            meanfreq[j] = numpy.mean(freq)
            meanfreq2[j] = numpy.mean(freq2)
        meanmeanfreq[kk] = numpy.mean(meanfreq)
        stdmeanfreq[kk] = numpy.std(meanfreq)
        meanmeanfreq2[kk] = numpy.mean(meanfreq2)
        stdmeanfreq2[kk] = numpy.std(meanfreq2)
        meanf[kk] = numpy.mean(freq)
        stdfreq[kk] = numpy.std(freq)
        meanf2[kk] = numpy.mean(freq2)
        stdfreq2[kk] = numpy.std(freq2)
    subplot(2,2,kplot)
    title(['SNR =',num2str(SNR), 'dB'])
    plot(f0v-f00*0,meanmeanfreq,'-+',f0v-f00*0,meanmeanfreq2,'-d')
    xlabel('f''_s (Hz)')
    ylabel ('mean(\Delta_f) (Hz)')
    legend( 'variable','fixed')
    title(['SNR =',num2str(SNR), 'dB'])
    hold on
    drawnow
    figure(32)
    subplot(2,2,kplot)
    title(['SNR =',num2str(SNR), 'dB'])
    %subplot(212)
    plot(f0v-f00*0,stdmeanfreq,'-+',f0v-f00*0,stdmeanfreq2,'-d')
    drawnow
    legend( 'variable','fixed')
    xlabel('f''_s (Hz)')
    ylabel ('STD(\Delta_f)(Hz)')
    title(['SNR =',num2str(SNR), 'dB'])


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


def manual_ft_2(k):
    global signal
    global koef
    global cnt
    global dft
    out = numpy.multiply(signal, numpy.exp(numpy.multiply(koef, k)))
    retval = numpy.sum(out)
    print(k)
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
    dft_transform(1, 150000, tau, f)


if __name__ == "__main__":
    main(sys.argv[1:])
