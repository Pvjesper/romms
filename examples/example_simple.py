import load_dataset as ld
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as splin

def deconvolve_perfect_sweep(sig, pseq):
    assert sig.ndim == 2
    assert sig.shape[-1] == pseq.shape[-1]
    if pseq.ndim == 2:
        pseq = np.squeeze(pseq, axis=0)

    pseq = pseq / np.sum(pseq**2)

    p_n = np.flip(np.roll(pseq, -1))
    p_n_reverse = np.concatenate((np.array([0]), np.flip(p_n[1:])))
    rir_est = splin.matmul_toeplitz((p_n, p_n_reverse), sig.T)
    return rir_est.T

# Load dataset
dataset_folder = "c:/path/to/romms_dataset"
info, pos, signals, rir = ld.load(room="b", seq_len_ms=500, max_freq=2000, speed = "slow", downsampled = True, alt_array=False, dataset_folder=dataset_folder)
ld.plot_pos(info, pos, show = False)

# Deconvolving gives same result as using the RIR directly
rir_deconvolved_array0 = deconvolve_perfect_sweep(signals["mic_array0"], signals["loudspeaker_array0"])
rir_deconvolved_array1 = deconvolve_perfect_sweep(signals["mic_array1"], signals["loudspeaker_array1"])

fig, axes = plt.subplots(2,1, figsize=(12,9))
axes[0].plot(rir[5,:], linewidth=2, label="RIR from dataset") # microphone 5 = array0 mic 5
axes[0].plot(rir_deconvolved_array0[5,:], linewidth=1, linestyle="dashed", label="Deconvolved RIR")
axes[1].plot(rir[35,:], linewidth=2, label="RIR from dataset") # microphone 35 = array1 mic 5
axes[1].plot(rir_deconvolved_array1[5,:], linewidth=1, linestyle="dashed", label="Deconvolved RIR")
for i, ax in enumerate(axes):
    ax.set_title(f"Microphone {i*30 + 5}")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid()
plt.show()