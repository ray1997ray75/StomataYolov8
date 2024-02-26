

'''
The Wiener filter is a way to restore a blurred image. Let's suppose that the PSF is a real and symmetric signal, 
a power spectrum of the original true image and noise are not known, then a simplified 
Wiener formula is: Hw = H / abs(H)^2 + 1 / SNR
'''

import numpy as np
import cv2

def wiener_filter(img, kernel, noise_var):
    kernel /= np.sum(kernel)
    dummy, img_noise = cv2.cartToPolar(cv2.merge((np.zeros(img.shape[:2]), np.zeros(img.shape[:2]))), cv2.merge((np.real(cv2.dft(img)), np.imag(cv2.dft(img)))), angleInDegrees=True)
    img_noise = img_noise + noise_var
    img_noise = cv2.merge((np.cos(np.deg2rad(img_noise)), np.sin(np.deg2rad(img_noise))))
    img_filtered = cv2.idft(cv2.merge((img_noise[:, :, 0] * kernel / (kernel ** 2 + noise_var), img_noise[:, :, 1] * kernel / (kernel ** 2 + noise_var))), flags=cv2.DFT_REAL_OUTPUT)
    return img_filtered

def wienerFilter(input,PSF , eps, K = 0.-1): ## Winener filter , K =0.01
    fftImg = np.fft.fft2(input)
    fftPSF = np.fft.fft2(PSF) + eps
    fftWiener = np.conj(fftPSF) / (np.abs(fftPSF) ** 2 + K)

# Load the image
image = cv2.imread('./T22/images/test/T22L600_W1_A4_R4_P231_g136_3.jpg', cv2.IMREAD_GRAYSCALE)

# Define the Wiener filter parameters (kernel size and noise variance)
kernel_size = 5
noise_variance = 0.01

# Generate the kernel (you can also use cv2.getGaussianKernel())
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)

# Apply the Wiener filter
filtered_image = wiener_filter(image, kernel, noise_variance)

# Convert to uint8 and display the result
filtered_image = np.uint8(filtered_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


