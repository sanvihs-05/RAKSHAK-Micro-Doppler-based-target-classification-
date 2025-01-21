import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.signal as signal

def generate_random_target_parameters(num_targets=3):
    """
    Generate random parameters for multiple moving targets
    """
    targets = []
    for _ in range(num_targets):
        target = {
            'velocity': np.random.uniform(10, 50),  # Random velocity between 10-50 m/s
            'frequency': np.random.uniform(20, 150),  # Random frequency between 20-150 Hz
            'amplitude': np.random.uniform(0.2, 1.0),  # Random amplitude between 0.2-1.0
            'phase': np.random.uniform(0, 2*np.pi)    # Random initial phase
        }
        targets.append(target)
    return targets

def generate_micro_doppler_signal(duration=2.0, fs=1000):
    """
    Generate a complex micro-Doppler signal with multiple random components
    
    Parameters:
    duration (float): Signal duration in seconds
    fs (int): Sampling frequency in Hz
    """
    t = np.linspace(0, duration, int(fs*duration))
    
    # Generate random carrier frequency
    carrier_freq = np.random.uniform(200, 300)
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    # Generate random targets
    targets = generate_random_target_parameters()
    signal = np.zeros_like(t, dtype=complex)
    
    # Add contribution from each target
    for target in targets:
        # Create time-varying Doppler shift
        doppler_shift = target['velocity'] * np.sin(2*np.pi*target['frequency']*t + target['phase'])
        target_signal = target['amplitude'] * np.exp(1j * 2 * np.pi * doppler_shift * t)
        signal += target_signal
    
    # Add random micro-motions (vibrations, rotations)
    num_micro_motions = np.random.randint(2, 5)
    for _ in range(num_micro_motions):
        micro_freq = np.random.uniform(5, 15)
        micro_amp = np.random.uniform(0.1, 0.3)
        micro_phase = np.random.uniform(0, 2*np.pi)
        signal += micro_amp * np.exp(1j * 2 * np.pi * micro_freq * t + micro_phase)
    
    # Add some noise
    noise_level = 0.1
    noise = noise_level * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    
    return (carrier * signal + noise), t

def compute_micro_doppler_signature(radar_data, fs=1000):
    """
    Compute micro-Doppler signature using STFT with random window parameters
    """
    # Random window parameters
    nperseg = np.random.choice([256, 512, 1024])  # Random window length
    overlap_ratio = np.random.uniform(0.6, 0.8)    # Random overlap ratio
    noverlap = int(nperseg * overlap_ratio)
    nfft = nperseg * 2  # FFT size
    
    # Create window
    window = signal.windows.hamming(nperseg)
    
    # Compute STFT
    f, t, Sxx = stft(radar_data, fs=fs, window=window,
                     nperseg=nperseg, noverlap=noverlap,
                     nfft=nfft)
    
    # Convert to power spectrogram in dB
    Sxx_db = 10 * np.log10(np.abs(Sxx)**2)
    
    return t, f, Sxx_db, nperseg, overlap_ratio

def plot_micro_doppler_signature(t, f, Sxx_db, title=''):
    """
    Plot the micro-Doppler signature with enhanced visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Normalize the spectrogram for better visualization
    Sxx_norm = Sxx_db - np.min(Sxx_db)
    Sxx_norm = Sxx_norm / np.max(Sxx_norm)
    
    plt.pcolormesh(t, f, Sxx_norm, shading='gouraud', cmap='jet')
    plt.colorbar(label='Normalized Power (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'Micro-Doppler Signature: {title}')
    
    # Set frequency limits to focus on the region of interest
    max_freq = np.max(np.abs(f))
    plt.ylim(-max_freq/2, max_freq/2)
    plt.grid(True, alpha=0.3)
    plt.show()

# Generate multiple random signatures
for i in range(3):
    # Random duration and sampling frequency
    duration = np.random.uniform(1.0, 5.0)
    fs = np.random.choice([1000, 2000, 10000])
    
    # Generate signal
    radar_signal, time = generate_micro_doppler_signal(duration, fs)
    
    # Compute signature
    t, f, Sxx_db, nperseg, overlap = compute_micro_doppler_signature(radar_signal, fs)
    
    # Plot with parameters in title
    title = f'Duration: {duration:.1f}s, Fs: {fs}Hz\nWindow: {nperseg}, Overlap: {overlap:.2f}'
    plot_micro_doppler_signature(t, f, Sxx_db, title)
    
    # Print parameters
    print(f"\nSignature {i+1} Parameters:")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Sampling Frequency: {fs} Hz")
    print(f"Window Length: {nperseg} points")
    print(f"Overlap Ratio: {overlap:.2f}")
    print("-" * 50)