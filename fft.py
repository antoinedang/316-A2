import sys
import numpy as np
import math
import os
import cv2
import time
import matplotlib.pyplot as plt

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
    
def discreteFourierTransform2D_Inner(x, k, l):
    return discreteFourierTransform(discreteFourierTransform(x, l), k)

def discreteFourierTransform2D(x, show_progress=True):
    out = np.zeros((len(x), len(x[0])), dtype=np.complex_)
    for l in range(len(x)):
        if show_progress:
            progress = 100 * (l*len(x[0])) / (len(x)*len(x[0]))
            print("{}%".format(progress))
        for k in range(len(x[0])):
            out[l][k] = discreteFourierTransform2D_Inner(x, k, l)
    return out

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

def inverseFastFourierTransform1D(X):
    x = []
    for n in range(len(X)):
        xn = inverseFastFourierTransform(X, n)
        x.append(xn)
    return np.array(x)

def inverseFastFourierTransform2D_Inner(X, m, n):
    return inverseFastFourierTransform(inverseFastFourierTransform(X, n), m)

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
    our_dft = discreteFourierTransform2D(test_matrix, show_progress=False)
    if not np.allclose(np_fft, our_dft):
        print("FAILED TEST\tour 2D DFT != numpy 2D FFT.")
        return False
    original = inverseFastFourierTransform2D(np_fft)
    np_original = np.fft.ifft2(np_fft)
    if not np.allclose(original, np_original):
        print("FAILED TEST\tour 2D IFFT != numpy 2D IFFT.")
        return False
    return True

def denoisingExperiments(img):
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
    high_freq_threshold = 0.85
    denoised_fft = fft_img.copy()
    rows, cols = denoised_fft.shape
    denoised_fft[int(rows*(0.5-high_freq_threshold/2)):int(rows*(0.5+high_freq_threshold/2)), :] = complex(0,0)
    denoised_fft[:, int(cols*(0.5-high_freq_threshold/2)):int(cols*(0.5+high_freq_threshold/2))] = complex(0,0)
    #remove borders at corners to remove lower frequencies
    low_freq_threshold = 1-3/512
    rows, cols = denoised_fft.shape
    denoised_fft[int(rows*(0.5+low_freq_threshold/2)):, :int(cols*(0.5-low_freq_threshold/2))] = complex(0,0)
    denoised_fft[:int(rows*(0.5-low_freq_threshold/2)), :int(cols*(0.5-low_freq_threshold/2))] = complex(0,0)
    denoised_fft[int(rows*(0.5+low_freq_threshold/2)):, int(cols*(0.5+low_freq_threshold/2)):] = complex(0,0)
    denoised_fft[:int(rows*(0.5-low_freq_threshold/2)), int(cols*(0.5+low_freq_threshold/2)):] = complex(0,0)
    #combine high and low freqs, apply inverse and normalization to display
    denoised_img = np.fft.ifft2(denoised_fft).real
    denoised_img_normalized = cv2.normalize(denoised_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display = np.concatenate((img, denoised_img_normalized), axis=1)
    cv2.imshow("denoised (removed high frequencies low + high frequencies)", display)
    cv2.waitKey(0)
    exit()

def sparsifyMatrix(matrix):
    sparse_matrix = [matrix.shape]
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if matrix[x][y] == 0: continue
            sparse_matrix.append((x,y,matrix[x][y]))
    return sparse_matrix

def sparseToCsv(sparse_matrix):
    out = str(sparse_matrix[0][0]) + "," + str(sparse_matrix[0][1]) + "\n"
    for x,y,val in sparse_matrix[1:]:
        out += str(x) + "," + str(y) + "," + str(val) + "\n"
    return out

pwd = os.path.realpath(os.path.dirname(__file__))

def saveToFile(data, file):
    with open(pwd + '/' + file, "w+") as f:
        f.write(data)

def zeroOutSmallestValues(fft_img, fractionToKeep):
    flattened = np.reshape(fft_img, (-1))
    new_flattened = np.zeros(flattened.shape, dtype=np.complex_)
    N = int(fractionToKeep*len(flattened))
    ind = np.argpartition(flattened, -N)[-N:]
    topN = flattened[ind]
    new_flattened[ind] = topN
    return np.reshape(new_flattened, fft_img.shape)

def zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep):
    high_freq_threshold = 0.8 # keep 10% border around image edges
    low_fft = fft_img.copy()
    rows, cols = low_fft.shape
    low_fft[int(rows*(0.5-high_freq_threshold/2)):int(rows*(0.5+high_freq_threshold/2)), :] = complex(0,0)
    low_fft[:, int(cols*(0.5-high_freq_threshold/2)):int(cols*(0.5+high_freq_threshold/2))] = complex(0,0)
    lowFrequencyCoefficientsKept = np.count_nonzero(low_fft)

    flattened = np.reshape(fft_img, (-1))
    new_flattened = np.zeros(flattened.shape, dtype=np.complex_)
    N = int(fractionToKeep*len(flattened))-lowFrequencyCoefficientsKept
    ind = np.argpartition(flattened, -N)[-N:]
    topN = flattened[ind]
    new_flattened[ind] = topN
    return np.reshape(new_flattened, fft_img.shape) + low_fft

def clipExtremeValues(img, deviation_range=2):
    std = np.std(img)
    mean = np.mean(img)
    img = np.where(img < mean + deviation_range*std, img, mean + deviation_range*std)
    img = np.where(img > mean - deviation_range*std, img, mean - deviation_range*std)
    return img

def compressionExperiments(img):
    fft_img = np.fft.fft2(img)
    #take FFT of image to compress it, display 6 different levels of compression from 0 to 95%

    #COMPRESSION METHOD 1: only keep top percent largest coefficients
    compressed_fft1 = zeroOutSmallestValues(fft_img, fractionToKeep=0.80)
    compressed_fft2 = zeroOutSmallestValues(fft_img, fractionToKeep=0.60)
    compressed_fft3 = zeroOutSmallestValues(fft_img, fractionToKeep=0.50)
    compressed_fft4 = zeroOutSmallestValues(fft_img, fractionToKeep=0.20)
    compressed_fft5 = zeroOutSmallestValues(fft_img, fractionToKeep=0.05)
    compressed_img1 = np.fft.ifft2(compressed_fft1).real
    compressed_img2 = np.fft.ifft2(compressed_fft1).real
    compressed_img3 = np.fft.ifft2(compressed_fft1).real
    compressed_img4 = np.fft.ifft2(compressed_fft1).real
    compressed_img5 = np.fft.ifft2(compressed_fft1).real
    
    compressed_img1 = cv2.normalize(compressed_img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img2 = cv2.normalize(compressed_img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img3 = cv2.normalize(compressed_img3, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img4 = cv2.normalize(compressed_img4, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img5 = cv2.normalize(compressed_img5, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    display_1 = np.concatenate((img, compressed_img1), axis=1)
    display_1 = np.concatenate((display_1, compressed_img2), axis=1)
    display_2 = np.concatenate((compressed_img3, compressed_img4), axis=1)
    display_2 = np.concatenate((display_2, compressed_img5), axis=1)
    display = np.concatenate((display_1, display_2), axis=0)
    displaySmaller = cv2.resize(display,(int(0.7*display.shape[1]),int(0.7*display.shape[0])))
    cv2.imshow('uncompressed, 20{} compressed, 40{} compressed, 60{} compressed, 80{} compressed, 95{} compressed (left to right, top to bottom)'.format('%','%','%','%','%','%'), displaySmaller)
    cv2.waitKey(0)
    
    #COMPRESSION METHOD 2: keep all low frequencises and top percent of largest coefficients
    compressed_fft1 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.80)
    compressed_fft2 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.60)
    compressed_fft3 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.50)
    compressed_fft4 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.20)
    compressed_fft5 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.05)
    compressed_img1 = np.fft.ifft2(compressed_fft1).real
    compressed_img2 = np.fft.ifft2(compressed_fft1).real
    compressed_img3 = np.fft.ifft2(compressed_fft1).real
    compressed_img4 = np.fft.ifft2(compressed_fft1).real
    compressed_img5 = np.fft.ifft2(compressed_fft1).real

    compressed_img1 = cv2.normalize(compressed_img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img2 = cv2.normalize(compressed_img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img3 = cv2.normalize(compressed_img3, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img4 = cv2.normalize(compressed_img4, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img5 = cv2.normalize(compressed_img5, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    display_1 = np.concatenate((img, compressed_img1), axis=1)
    display_1 = np.concatenate((display_1, compressed_img2), axis=1)
    display_2 = np.concatenate((compressed_img3, compressed_img4), axis=1)
    display_2 = np.concatenate((display_2, compressed_img5), axis=1)
    display = np.concatenate((display_1, display_2), axis=0)
    displaySmaller = cv2.resize(display,(int(0.7*display.shape[1]),int(0.7*display.shape[0])))
    cv2.imshow('uncompressed, 20{} compressed, 40{} compressed, 60{} compressed, 80{} compressed, 95{} compressed (left to right, top to bottom)'.format('%','%','%','%','%','%'), displaySmaller)
    cv2.waitKey(0)

    #COMPRESSION METHOD 3: compression method 2 + additional normalization of image using mean and standard deviation 
    compressed_fft1 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.80)
    compressed_fft2 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.60)
    compressed_fft3 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.50)
    compressed_fft4 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.20)
    compressed_fft5 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.05)
    compressed_img1 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img2 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img3 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img4 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img5 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)

    compressed_img1 = cv2.normalize(compressed_img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img2 = cv2.normalize(compressed_img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img3 = cv2.normalize(compressed_img3, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img4 = cv2.normalize(compressed_img4, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img5 = cv2.normalize(compressed_img5, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    display_1 = np.concatenate((img, compressed_img1), axis=1)
    display_1 = np.concatenate((display_1, compressed_img2), axis=1)
    display_2 = np.concatenate((compressed_img3, compressed_img4), axis=1)
    display_2 = np.concatenate((display_2, compressed_img5), axis=1)
    display = np.concatenate((display_1, display_2), axis=0)
    displaySmaller = cv2.resize(display,(int(0.7*display.shape[1]),int(0.7*display.shape[0])))
    cv2.imshow('uncompressed, 20{} compressed, 40{} compressed, 60{} compressed, 80{} compressed, 95{} compressed (left to right, top to bottom)'.format('%','%','%','%','%','%'), displaySmaller)
    cv2.waitKey(0)
    exit()

if not testFTs(): exit()

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

if not os.path.isfile(pwd + "/" + inputFile):
    print("ERROR\tIncorrect input: file {} does not exist in the current directory.".format(inputFile))
    exit()

img = cv2.imread(pwd + "/" + inputFile, 0)
img = cv2.resize(img, (nearestPowerOf2(img.shape[0]), nearestPowerOf2(img.shape[1])))

#denoisingExperiments(img)

if mode == 1:
    #image converted to FFT and displayed alongside original
    fft_img = fastFourierTransform2D(img).real
    logged = np.log1p(fft_img)
    normalized_fft = cv2.normalize(logged, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display = np.concatenate((img, normalized_fft), axis=1)
    cv2.imshow("log scaled fft (our implementation)", display)
    cv2.waitKey(0)
    # WITH NUMPY
    #fft_img = np.fft.fft2(img).real
    #logged = np.log1p(fft_img)
    #normalized_fft = cv2.normalize(logged, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #display = np.concatenate((img, normalized_fft), axis=1)
    #cv2.imshow("log scaled fft (with numpy)", display)
    #cv2.waitKey(0)

elif mode == 2:
    #denoise image by applying GGT, truncating high frequencies, and displaying it alongside original
    fft_img = fastFourierTransform2D(img)
    #keep only corners of image to remove high frequencies (all central parts of image are the highest frequencies)
    high_freq_threshold = 0.85
    denoised_fft = fft_img.copy()
    rows, cols = denoised_fft.shape
    denoised_fft[int(rows*(0.5-high_freq_threshold/2)):int(rows*(0.5+high_freq_threshold/2)), :] = complex(0,0)
    denoised_fft[:, int(cols*(0.5-high_freq_threshold/2)):int(cols*(0.5+high_freq_threshold/2))] = complex(0,0)
    #remove borders at corners to remove lower frequencies
    low_freq_threshold = 1-3/512
    rows, cols = denoised_fft.shape
    denoised_fft[int(rows*(0.5+low_freq_threshold/2)):, :int(cols*(0.5-low_freq_threshold/2))] = complex(0,0)
    denoised_fft[:int(rows*(0.5-low_freq_threshold/2)), :int(cols*(0.5-low_freq_threshold/2))] = complex(0,0)
    denoised_fft[int(rows*(0.5+low_freq_threshold/2)):, int(cols*(0.5+low_freq_threshold/2)):] = complex(0,0)
    denoised_fft[:int(rows*(0.5-low_freq_threshold/2)), int(cols*(0.5+low_freq_threshold/2)):] = complex(0,0)
    #combine high and low freqs, apply inverse and normalization to display
    denoised_img = inverseFastFourierTransform2D(denoised_fft).real
    denoised_img_normalized = cv2.normalize(denoised_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display = np.concatenate((img, denoised_img_normalized), axis=1)
    cv2.imshow("denoised (removed high frequencies low + high frequencies)", display)
    cv2.waitKey(0)


elif mode == 3:
    fft_img = np.fft.fft2(img)
    #take FFT of image to compress it, display 6 different levels of compression from 0 to 95%

    #COMPRESSION METHOD: keep all low frequencises and top percent of largest coefficients + remove extreme values from resulting image
    compressed_fft1 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.80)
    compressed_fft2 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.60)
    compressed_fft3 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.50)
    compressed_fft4 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.20)
    compressed_fft5 = zeroOutSmallestValuesKeepLowFrequencies(fft_img, fractionToKeep=0.05)
    compressed_img1 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img2 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img3 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img4 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)
    compressed_img5 = clipExtremeValues(np.fft.ifft2(compressed_fft1).real)

    compressed_img1 = cv2.normalize(compressed_img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img2 = cv2.normalize(compressed_img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img3 = cv2.normalize(compressed_img3, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img4 = cv2.normalize(compressed_img4, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    compressed_img5 = cv2.normalize(compressed_img5, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    display_1 = np.concatenate((img, compressed_img1), axis=1)
    display_1 = np.concatenate((display_1, compressed_img2), axis=1)
    display_2 = np.concatenate((compressed_img3, compressed_img4), axis=1)
    display_2 = np.concatenate((display_2, compressed_img5), axis=1)
    display = np.concatenate((display_1, display_2), axis=0)
    displaySmaller = cv2.resize(display,(int(0.7*display.shape[1]),int(0.7*display.shape[0])))
    cv2.imshow('uncompressed, 20{} compressed, 40{} compressed, 60{} compressed, 80{} compressed, 95{} compressed (left to right, top to bottom)'.format('%','%','%','%','%','%'), displaySmaller)
    cv2.waitKey(0)
    #for each compression save a sparse version of the compressed FFT to .txt or .csv file
    sparse_compressed_fft_img = sparsifyMatrix(fft_img)
    csv_data = sparseToCsv(sparse_compressed_fft_img) 
    saveToFile(csv_data, 'uncompressed.csv')
    sparse_compressed_fft1 = sparsifyMatrix(compressed_fft1)
    csv_data = sparseToCsv(sparse_compressed_fft1) 
    saveToFile(csv_data, 'compressed20percent.csv')
    sparse_compressed_fft2 = sparsifyMatrix(compressed_fft2)
    csv_data = sparseToCsv(sparse_compressed_fft2) 
    saveToFile(csv_data, 'compressed40percent.csv')
    sparse_compressed_fft3 = sparsifyMatrix(compressed_fft3)
    csv_data = sparseToCsv(sparse_compressed_fft3) 
    saveToFile(csv_data, 'compressed60percent.csv')
    sparse_compressed_fft4 = sparsifyMatrix(compressed_fft4)
    csv_data = sparseToCsv(sparse_compressed_fft4) 
    saveToFile(csv_data, 'compressed80percent.csv')
    sparse_compressed_fft5 = sparsifyMatrix(compressed_fft5)
    csv_data = sparseToCsv(sparse_compressed_fft5) 
    saveToFile(csv_data, 'compressed95percent.csv')
    #print number of nonzero fourier coefficients in each of the 6 images
    print("Original image nonzero coefficients: {}".format(np.count_nonzero(fft_img)))
    print("Compressed image (20%) nonzero coefficients: {}".format(np.count_nonzero(compressed_fft1)))
    print("Compressed image (40%) nonzero coefficients: {}".format(np.count_nonzero(compressed_fft2)))
    print("Compressed image (60%) nonzero coefficients: {}".format(np.count_nonzero(compressed_fft3)))
    print("Compressed image (80%) nonzero coefficients: {}".format(np.count_nonzero(compressed_fft4)))
    print("Compressed image (95%) nonzero coefficients: {}".format(np.count_nonzero(compressed_fft5)))
    #justify scheme

elif mode == 4:
    #plot runtime graphs
    number_runs = 10
    problem_sizes = [2**5, 2**6, 2**7, 2**8, 2**9]
    runtimes_DFT = np.zeros((number_runs, len(problem_sizes)))
    runtimes_FFT = np.zeros((number_runs, len(problem_sizes)))
    for n in range(number_runs):
        for p in range(len(problem_sizes)):
            progress = 100 * (n*len(problem_sizes) + p)/(len(problem_sizes)*number_runs)
            print(str(progress) + "%")
            test_matrix = np.random.rand(problem_sizes[p], problem_sizes[p])
            start_time = time.time()
            discreteFourierTransform2D(test_matrix, show_progress=False)
            end_time = time.time()
            runtimes_DFT[n][p] = end_time-start_time
            start_time = time.time()
            fastFourierTransform2D(test_matrix, show_progress=False)
            end_time = time.time()
            runtimes_FFT[n][p] = end_time-start_time
    means_DFT = np.mean(runtimes_DFT, axis=0)
    stds_DFT = np.std(runtimes_DFT, axis=0)
    means_FFT = np.mean(runtimes_FFT, axis=0)
    stds_FFT = np.std(runtimes_FFT, axis=0)
    plt.figure()
    plt.plot(problem_sizes, means_DFT)
    plt.errorbar(problem_sizes, means_DFT, 2*stds_DFT)
    plt.xlabel('Problem Size (size of one side of square matrix)')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Naive Fourier Transform Mean Runtime')
    plt.show()
    plt.figure()
    plt.plot(problem_sizes, means_FFT)
    plt.errorbar(problem_sizes, means_FFT, 2*stds_FFT)
    plt.xlabel('Problem Size (size of one side of square matrix)')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Fast Fourier Transform Mean Runtime')
    plt.show()



