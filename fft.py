import sys
import numpy as np
import math
import os
import cv2

def verifyValidity(name, value):
    if name == "mode":
        try:
            if int(value) > 4 or int(value) < 1:
                raise Exception
        except:
            print("ERROR\tIncorrect mode argument: mode must be one of [1,2,3,4].")
            exit()

#WORKS
def discreteFourierTransform(x, k):
    Xk = complex(0,0)
    for n in range(len(x)):
        Xk += x[n]*np.exp(math.pi*complex(0,-2)*k*n/len(x))
    return Xk

#WORKS
def discreteFourierTransform1D(x):
    X = []
    for k in range(len(x)):
        Xk = discreteFourierTransform(x, k)
        X.append(Xk)
    return np.array(X)

#WORKS
fft_threshold = 16
def fastFourierTransform(x, k):
    if len(x) <= fft_threshold:
        return(discreteFourierTransform(x, k))
    x_even = [x[i] for i in range(len(x)) if i%2==0]
    x_odd = [x[i] for i in range(len(x)) if i%2!=0]
    Xk = fastFourierTransform(x_even, k) + np.exp(-2*math.pi*complex(0,1)*k/len(x))*fastFourierTransform(x_odd, k)
    return Xk

#WORKS
def fastFourierTransform1D(x):
    X = []
    for k in range(len(x)):
        Xk = fastFourierTransform(x, k)
        X.append(Xk)
    return np.array(X)

#WORKS
def fastFourierTransform2D_Inner(x, k, l):
    return fastFourierTransform(fastFourierTransform(x, l), k)

#WORKS
def fastFourierTransform2D(x):
    out = np.zeros((len(x), len(x[0])), dtype=np.complex_)
    for l in range(len(x)):
        for k in range(len(x[0])):
            progress = 100 * (k + (l*len(x[0]))) / (len(x)*len(x[0]))
            print("{}%".format(progress))
            out[l][k] = fastFourierTransform2D_Inner(x, k, l)
    return out

#WORKS
def inverseDiscreteFourierTransform(X, n, N=None):
    if N == None:
        N = len(X)
    xn = complex(0,0)
    for k in range(len(X)):
        xn += X[k]*np.exp(math.pi*complex(0,2)*k*n/len(X))
    return (xn / N)

#WORKS
def inverseFastFourierTransform(X, n, N=None):
    if N == None:
        N = len(X)
    if len(X) <= fft_threshold:
        return(inverseDiscreteFourierTransform(X, n, N))
    x_even = [X[i] for i in range(len(X)) if i%2==0]
    x_odd = [X[i] for i in range(len(X)) if i%2!=0]
    xn = inverseFastFourierTransform(x_even, n, N) + np.exp(2*math.pi*complex(0,1)*n/len(X))*inverseFastFourierTransform(x_odd, n, N)
    return xn

#WORKS
def inverseFastFourierTransform1D(X):
    x = []
    for n in range(len(X)):
        xn = inverseFastFourierTransform(X, n)
        x.append(xn)
    return np.array(x)

#WORKS
def inverseFastFourierTransform2D_Inner(X, m, n):
    return inverseFastFourierTransform(inverseFastFourierTransform(X, n), m)

#WORKS
def inverseFastFourierTransform2D(X):
    out = np.zeros((len(X), len(X[0])), dtype=np.complex_)
    for n in range(len(X)):
        for m in range(len(X[0])):
            out[n][m] = inverseFastFourierTransform2D_Inner(X, m, n)
    return out

def nearestPowerOf2(x):
    i = 1
    while i*2 < x:
        i *= 2
    if i*2 - x < x - i:
        return i*2
    else:
        return i 

arguments = {}
nextArg = None
for arg in sys.argv[1:]:
    if nextArg != None:
        verifyValidity(nextArg, arg)
        arguments[nextArg] = arg
        nextArg = None
        continue

    if arg == '-m':
        if len(arguments.keys()) != 0:
            print("ERROR\tIncorrect input syntax: Mode argument must be the first argument.")
            exit()
        nextArg = "mode"
    elif arg == '-i':
        nextArg = "input"
    else:
        print("ERROR\tIncorrect input syntax: Only -i and -m arguments accepted.")
        exit()

if nextArg != None:
    print("ERROR\tIncorrect input syntax: missing value following {} argument.".format(nextArg))
    exit()

mode = int(arguments.get("mode", 1))
inputFile = arguments.get("input", 'moonlanding.png')

print("Mode: {}".format(mode))
print("Input: {}".format(inputFile))

pwd = os.path.realpath(os.path.dirname(__file__))

if not os.path.isfile(pwd + "/" + inputFile):
    print("ERROR\tIncorrect input: file {} does not exist in the current directory.".format(inputFile))
    exit()

img = cv2.imread(pwd + "/" + inputFile)
#nearestPowerOf2(img.shape[0]), nearestPowerOf2(img.shape[1])
img = cv2.resize(img, (32,32))
b,g,r = cv2.split(img)

if mode == 1:
    #image converted to FFT and displayed alongside original
    r_fft = fastFourierTransform2D(r).real
    print("transformed r")
    g_fft = fastFourierTransform2D(g).real
    print("transformed g")
    b_fft = fastFourierTransform2D(b).real
    print("transformed b")
    fft_img = cv2.merge((b_fft, g_fft, r_fft))
    logged = np.uint8(np.log1p(fft_img))
    normalized_fft = cv2.normalize(logged, None, 0, 255, cv2.NORM_MINMAX, dtype =cv2.CV_8U)
    print(fft_img)
    #combined_imgs = np.concatenate((img, normalized_fft), axis=1)
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.imshow("log-scaled fft", normalized_fft)
    cv2.waitKey(0)

elif mode == 2:
    #denoise image by applying GGT, truncating high frequencies, and displaying it alongside original
    pass
elif mode == 3:
    #take FFT of image to compress it, display 6 different levels of compression from 0 to 95%
    #for each compression save to .txt or .csv file
    #print nonzero fourier coefficients in each of the 6 images
    #justify scheme
    pass
elif mode == 4:
    #plot runtime graphs
    pass