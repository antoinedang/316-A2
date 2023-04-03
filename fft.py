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

def discreteFourierTransform(x, k):
    Xk = complex(0,0)
    for n in range(len(x)):
        Xk += x[n]*np.exp(math.pi*complex(0,-2)*k*n/len(x))
    return Xk

def discreteFourierTransform1D(x):
    X = []
    for k in range(len(x)):
        Xk = discreteFourierTransform(x, k)
        X.append(Xk)
    return np.array(X)

fft_threshold = 8
def fastFourierTransform(x, k):
    if len(x) <= fft_threshold:
        return(discreteFourierTransform(x, k))
    x_even = [x[i] for i in range(len(x)) if i%2==0]
    x_odd = [x[i] for i in range(len(x)) if i%2!=0]
    Xk = fastFourierTransform(x_even, k) + np.exp(-2*math.pi*complex(0,1)*k/len(x))*fastFourierTransform(x_odd, k)
    return Xk

def fastFourierTransform1D(x):
    X = []
    for k in range(len(x)):
        Xk = fastFourierTransform(x, k)
        X.append(Xk)
    return np.array(X)

def fastFourierTransform2D_Inner(x, k, l):
    return fastFourierTransform(fastFourierTransform(x, l), k)

def fastFourierTransform2D(x, show_progress=True):
    out = np.zeros((len(x), len(x[0])), dtype=np.complex_)
    for l in range(len(x)):
        if show_progress:
            progress = 100 * (l*len(x[0])) / (len(x)*len(x[0]))
            print("{}%".format(progress))
        for k in range(len(x[0])):
            out[l][k] = fastFourierTransform2D_Inner(x, k, l)
    return out

def inverseDiscreteFourierTransform(X, n, N=None):
    if N == None:
        N = len(X)
    xn = complex(0,0)
    for k in range(len(X)):
        xn += X[k]*np.exp(math.pi*complex(0,2)*k*n/len(X))
    return (xn / N)

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

def testFTs():
    random_shape_w = 2**np.random.randint(0, 6)
    random_shape_h = 2**np.random.randint(0, 6)
    test_matrix = np.random.rand(random_shape_w, random_shape_h)
    for test_row in test_matrix:
        #TEST 1D for each row
        np_fft = np.fft.fft(test_row)
        our_dft = discreteFourierTransform1D(test_row)
        our_fft = fastFourierTransform1D(test_row)
        if not np.allclose(np_fft, our_dft):
            print("FAILED TEST\tour 1D DFT != numpy 1D FFT.")
            return False
        if not np.allclose(np_fft, our_fft):
            print("FAILED TEST\tour 1D FFT != numpy 1D FFT.")
            return False
        original = inverseFastFourierTransform1D(np_fft)
        np_original = np.fft.ifft(np_fft)
        if not np.allclose(original, np_original):
            print("FAILED TEST\tour 1D IFFT != numpy 1D IFFT.")
            return False
    #TEST 2D for matrix
    np_fft = np.fft.fft2(test_matrix)
    our_fft = fastFourierTransform2D(test_matrix, show_progress=False)
    if not np.allclose(np_fft, our_fft):
        print("FAILED TEST\tour 2D FFT != numpy 2D FFT.")
        return False
    original = inverseFastFourierTransform2D(np_fft)
    np_original = np.fft.ifft2(np_fft)
    if not np.allclose(original, np_original):
        print("FAILED TEST\tour 2D IFFT != numpy 2D IFFT.")
        return False
    return True

if testFTs(): print("ALL TESTS PASSED!")
else: exit()

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

img = cv2.imread(pwd + "/" + inputFile, 0)
img = cv2.resize(img, (nearestPowerOf2(img.shape[0]), nearestPowerOf2(img.shape[1])))

if mode == 1:
    #image converted to FFT and displayed alongside original
    fft_img = fastFourierTransform2D(img).real
    logged = np.log1p(fft_img)
    normalized_fft = cv2.normalize(logged, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display = np.concatenate((img, normalized_fft), axis=1)
    cv2.imshow("log scaled fft (our implementation)", display)
    cv2.waitKey(0)
    #WITH NUMPY
    #fft_img = np.fft.fft2(img).real
    #logged = np.log1p(fft_img)
    #normalized_fft = cv2.normalize(logged, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #display = np.concatenate((img, normalized_fft), axis=1)
    #cv2.imshow("log scaled fft (with numpy)", display)
    #cv2.waitKey(0)

elif mode == 2:
    #denoise image by applying GGT, truncating high frequencies, and displaying it alongside original
    fft_img = np.fft.fft2(img)

    #DENOISING METHOD 1 - removing low frequencies 
    #keep only central part of image to reduce low frequencies
    low_freq_threshold = 0.8
    denoised_fft = fft_img.copy()
    rows, cols = denoised_fft.shape
    denoised_fft[:, :int(cols*(0.5-low_freq_threshold/2))] = complex(0,0)
    denoised_fft[:, int(cols*(0.5+low_freq_threshold/2)):] = complex(0,0)
    denoised_fft[:int(rows*(0.5-low_freq_threshold/2)), :] = complex(0,0)
    denoised_fft[int(rows*(0.5+low_freq_threshold/2)):, :] = complex(0,0)
    #apply inverse and normalization to display
    denoised_img = np.fft.ifft2(denoised_fft).real
    denoised_img_normalized = cv2.normalize(denoised_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display = np.concatenate((img, denoised_img_normalized), axis=1)
    cv2.imshow("denoised (only high frequencies)", display)
    cv2.waitKey(0)

    #DENOISING METHOD 2 - removing high frequencies
    #keep only corners of image to remove high frequencies (all central parts of image axes are the highest frequencies)
    high_freq_threshold = 0.8
    denoised_fft = fft_img.copy()
    rows, cols = denoised_fft.shape
    denoised_fft[int(rows*(0.5-high_freq_threshold/2)):int(rows*(0.5+high_freq_threshold/2)), :] = complex(0,0)
    denoised_fft[:, int(cols*(0.5-high_freq_threshold/2)):int(cols*(0.5+high_freq_threshold/2))] = complex(0,0)
    #apply inverse and normalization to display
    denoised_img = np.fft.ifft2(denoised_fft).real
    denoised_img_normalized = cv2.normalize(denoised_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display = np.concatenate((img, denoised_img_normalized), axis=1)
    cv2.imshow("denoised (only low frequencies)", display)
    cv2.waitKey(0)

    #DENOISING METHOD 3 - removing low and high frequencies
    #keep only corners of image to remove high frequencies (all central parts of image are the highest frequencies)
    high_freq_threshold = 0.8
    denoised_fft = fft_img.copy()
    rows, cols = denoised_fft.shape
    denoised_fft[int(rows*(0.5-high_freq_threshold/2)):int(rows*(0.5+high_freq_threshold/2)), :] = complex(0,0)
    denoised_fft[:, int(cols*(0.5-high_freq_threshold/2)):int(cols*(0.5+high_freq_threshold/2))] = complex(0,0)
    denoised_fft_normalized = cv2.normalize(denoised_fft.imag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("filtered fft", denoised_fft_normalized)
    cv2.waitKey(0)
    #remove borders at corners to remove lower frequencies
    low_freq_threshold = 0.98
    rows, cols = denoised_fft.shape
    denoised_fft[int(rows*(0.5+low_freq_threshold/2)):, :int(cols*(0.5-low_freq_threshold/2))] = complex(0,0)
    denoised_fft[:int(rows*(0.5-low_freq_threshold/2)), :int(cols*(0.5-low_freq_threshold/2))] = complex(0,0)
    denoised_fft[int(rows*(0.5+low_freq_threshold/2)):, int(cols*(0.5+low_freq_threshold/2)):] = complex(0,0)
    denoised_fft[:int(rows*(0.5-low_freq_threshold/2)), int(cols*(0.5+low_freq_threshold/2)):] = complex(0,0)
    denoised_fft_normalized = cv2.normalize(denoised_fft.real, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("filtered fft", denoised_fft_normalized)
    cv2.waitKey(0)
    #combine high and low freqs, apply inverse and normalization to display
    denoised_img = np.fft.ifft2(denoised_fft).real
    denoised_img_normalized = cv2.normalize(denoised_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display = np.concatenate((img, denoised_img_normalized), axis=1)
    cv2.imshow("denoised (remove low + high frequencies)", display)
    cv2.waitKey(0)
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