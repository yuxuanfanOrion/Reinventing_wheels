import numpy as np
import matplotlib.pyplot as plt

def fft(x):
    """
    Compute the Fast Fourier Transform of an array.
    
    Args:
        x (numpy.ndarray): Input array.
        
    Returns:
        numpy.ndarray: FFT of the input array.
    """
    N = x.shape[0]
    if N <= 1:
        return x
    
    even = fft(x[0::2])
    odd = fft(x[1::2])
    
    T = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + T[:N // 2] * odd,
                           even + T[N // 2:] * odd])

def ifft(x):
    """
    Compute the Inverse Fast Fourier Transform of an array.
    
    Args:
        x (numpy.ndarray): Input array.
        
    Returns:
        numpy.ndarray: Inverse FFT of the input array.
    """
    x_conjugate = np.conjugate(x)
    y = fft(x_conjugate)
    return np.conjugate(y) / x.shape[0]

# Example usage:
if __name__ == "__main__":
    # Create a sample signal
    sample_rate = 1024  # Sampling frequency
    T = 1.0 / sample_rate  # Sampling interval
    L = 1024  # Length of signal
    t = np.linspace(0.0, L * T, L, endpoint=False)
    freq1 = 50  # Frequency of the first sine wave
    freq2 = 120  # Frequency of the second sine wave
    signal = 0.6 * np.sin(2 * np.pi * freq1 * t) + 0.4 * np.sin(2 * np.pi * freq2 * t)

    # Compute the FFT
    fft_result = fft(signal)

    # Compute the frequencies corresponding to the FFT result
    freqs = np.fft.fftfreq(L, T)

    # Plot the original signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title("Original Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # Plot the FFT result
    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.abs(fft_result))
    plt.title("FFT of the Signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()
