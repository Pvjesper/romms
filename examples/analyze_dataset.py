import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import pyroomacoustics as pra
import scipy.signal as spsig
import scipy.stats as spstats

import aspcore.fouriertransform as ft
import aspcore.filterdesign as fd
import aspcore.utilities as utils

import load_dataset as ld

def estimate_rt60(dataset_path, fig_folder):
    seq_len_ms = 1000
    max_freq = 4000
    sr = max_freq*2
    num_freq = utils.next_power_of_two(max_freq)
    rir_arrays = []
    for room_name in ["a" , "b", "c", "d"]:
        info, pos, signals, rir  = ld.load(room_name, seq_len_ms, max_freq, speed=None, downsampled=True, dataset_folder=dataset_path)
        rir_arrays.append(rir)

    rt60s_and_freqs = [measure_rt60(rir_array, sr) for rir_array in rir_arrays]
    rt60s = [a[0] for a in rt60s_and_freqs]
    freq_lims = [a[1] for a in rt60s_and_freqs]

    fig, ax = plt.subplots(1,1)
    for i, rt60 in enumerate(rt60s):
        rt60_mean = np.mean(rt60, axis=-1)
        ax.plot(rt60_mean, label=f"{i}")
        rt60_std = np.std(rt60, axis=-1)
        rt60_min = np.min(rt60, axis=-1)
        rt60_max = np.max(rt60, axis=-1)

        ax.fill_between(np.arange(rt60.shape[0]), rt60_mean - rt60_std, rt60_mean + rt60_std, alpha=0.15)
    ax.legend()
    utils.save_plot("pdf", fig_folder, "rt60")

    colors = ["green", "blue", "red", "yellow"]
    fig, ax = plt.subplots(1,1)
    ax.set_xscale('log', base=2)

    assert np.all([np.allclose(freq_lims[i], freq_lims[0]) for i in range(len(freq_lims))])

    for i, rt60 in enumerate(rt60s):
        rt60_mean = np.mean(rt60, axis=-1)
        rt60_std = np.std(rt60, axis=-1)

        x = freq_lims[0].reshape(-1)
        y = np.repeat(rt60_mean, 2)
        x = np.insert(x, np.arange(2, x.shape[-1], 2), np.nan)
        y = np.insert(y, np.arange(2, y.shape[-1], 2), np.nan)
        ax.plot(x,y, label=f"{i}", color=colors[i], linewidth=3)

        std_plus = rt60_mean + rt60_std
        std_minus = rt60_mean - rt60_std
        y_plus = np.repeat(std_plus, 2)
        y_minus = np.repeat(std_minus, 2)
        y_plus = np.insert(y_plus, np.arange(2, y_plus.shape[-1], 2), np.nan)
        y_minus = np.insert(y_minus, np.arange(2, y_minus.shape[-1], 2), np.nan)
        #ax.fill_between(x, y_minus, y_plus, label=f"{i}", color=colors[i], alpha=0.1)
        ax.plot(x, y_plus, label=f"{i}", color=colors[i], alpha=0.3)
        ax.plot(x, y_minus, label=f"{i}", color=colors[i], alpha=0.3)

        # num_bands = rt60.shape[0]
        # for b in range(num_bands):
        #     ax.plot(freq_lims[i][b,:], [rt60_mean[b], rt60_mean[b]], label=f"{i}", color=colors[i], linewidth=3)
        #     ax.fill_between(freq_lims[i][b,:], (rt60_mean[b] - rt60_std[b], rt60_mean[b] - rt60_std[b]), 
        #                                         (rt60_mean[b] + rt60_std[b], rt60_mean[b] + rt60_std[b]), label=f"{i}", color=colors[i], alpha=0.1)
    ax.legend()
    utils.save_plot("pdf", fig_folder, "rt60_intervals")

    # COMPUTE AVERAGE RT60
    rt60_summary = {}
    rt60_summary["mean"] = {}
    rt60_summary["std"] = {}
    rt60_summary["min"] = {}
    rt60_summary["max"] = {}

    #rt60_mean = {}
    #rt60_std = {}
    #rt60_min = {}
    #rt60_max = {}
    for room_name, rir in zip(["a", "b", "c", "d"], rir_arrays):
        pass
        rt60s = [pra.experimental.rt60.measure_rt60(rir[m,:], fs=sr, decay_db=20, plot = False, energy_thres=0.993) for m in range(pos["mic"].shape[0])]

        rt60_summary["mean"][room_name] = np.mean(rt60s, axis=0)
        rt60_summary["std"][room_name] = np.std(rt60s, axis=0)
        rt60_summary["min"][room_name] = np.min(rt60s, axis=0)
        rt60_summary["max"][room_name] = np.max(rt60s, axis=0)
    with open(fig_folder / "rt60_summary.json", "w") as f:
        json.dump(rt60_summary, f, indent=4)



def measure_rt60(rir, sr):
    """
    
    
    Parameters
    ----------
    rir : ndarray of shape (num_channels, rir_len)
        room impulse responses to measure RT60 for
    sr : int
        samplerate
    """
    rir_filtered, freq_lims = fd.filterbank_third_octave(rir, sr, min_freq = 40, plot=False)
    num_bands = rir_filtered.shape[0]
    num_mic = rir_filtered.shape[1]

    rt60s = np.zeros((num_bands, num_mic))

    #pra.experimental.rt60.measure_rt60(rir_filtered[0,0,:], sr, 40, True)
    for m in range(num_mic):
        for b in range(num_bands):
            rt60s[b,m] = pra.experimental.rt60.measure_rt60(rir_filtered[b,m,:], fs=sr, decay_db=20, plot = False, energy_thres=0.993)
    return rt60s, freq_lims




def _calc_speed(trajectory, sr):
    shift = 100
    diff = trajectory[:,shift:,:] - trajectory[:,:-shift,:]
    distance_per_shift = np.linalg.norm(diff, axis=-1)

    distance_per_second = distance_per_shift * sr / shift
    return distance_per_second

def measure_speed_of_microphone(dataset_path, fig_folder):
    seq_len_ms = 1000
    max_freq = 4000
    sr = max_freq*2
    num_freq = utils.next_power_of_two(max_freq)
    #calc_speed summary
    for speed_setting in ("fast", "slow"):
        speed_mean = []
        speed_median = []
        speed_std = []

        for room_name in ["a" , "b", "c", "d"]:
            if room_name == "c" and speed_setting == "slow":
                continue
            for seq_len_ms in [500, 1000]:
                for max_freq in [500, 1000, 2000, 4000]:

                    info, pos, signals, rir  = ld.load(room_name, seq_len_ms, max_freq, speed=speed_setting, downsampled=True, dataset_folder=dataset_path)
                    speed = _calc_speed(pos["mic_moving"], info["samplerate"])
                    
                    speed_mean.append(np.mean(speed, axis=-1))
                    speed_median.append(np.median(speed, axis=-1))
                    speed_std.append(np.std(speed, axis=-1))

        fig, ax = plt.subplots(1,1)
        ax.plot(speed_mean, label="mean")
        ax.plot(speed_median, label="median")
        ax.set_xlabel("'Run' index")
        ax.set_ylabel("Speed [m/s]")
        ax.legend()
        utils.set_basic_plot_look(ax)
        utils.save_plot("pdf", fig_folder, f"speed_summary_{speed_setting}")

        speed_mean_total = np.mean(np.array(speed_mean), axis=0)
        speed_std_total = np.std(np.array(speed_mean), axis=0)
        with open(fig_folder / f"speed_summary_{speed_setting}.json", "w") as f:
            json.dump({"mean": speed_mean_total.tolist(), "std": speed_std_total.tolist()}, f, indent=4)
        

    # plot speed
    
    seq_len_ms = 1000
    max_freq = 4000
    fig, ax = plt.subplots(1,1, figsize= (8,6))

    for speed_setting in ("fast", "slow"):
        for room_name in ["a" , "b", "c", "d"]:
            #if room_name == "c" and speed_setting == "slow":
            #    continue
            info, pos, signals, rir  = ld.load(room_name, seq_len_ms, max_freq, speed=speed_setting, downsampled=True, dataset_folder=dataset_path)
            speed = _calc_speed(pos["mic_moving"], info["samplerate"])

            ax.plot(np.linspace(0, speed.shape[-1] / (max_freq*2), speed.shape[-1]), speed[0,:], label=f"{room_name} {speed_setting}")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    utils.set_basic_plot_look(ax)
    utils.save_plot("pdf", fig_folder, "speed")

def plot_signal_levels(dataset_path, fig_folder):
    room_name = "a"
    seq_len_ms = 1000
    max_freq = 4000

    info, pos, signals, rir  = ld.load(room_name, seq_len_ms, max_freq, speed="slow", downsampled=True, dataset_folder=dataset_path)
    _, pos_noise, noise  = ld.load_noise(room_name, max_freq, speed="slow", downsampled=True, dataset_folder=dataset_path)

    # Plot energy levels for the stationary microphones
    fft_len = 2048
    num_tile = 25
    fig, ax = plt.subplots(1,1)
    f, pxx = spsig.welch(np.tile(signals["mic_array0"], (1,num_tile)), fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    f, pxx2 = spsig.welch(np.tile(signals["mic_array1"], (1,num_tile)), fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    pxx = np.concatenate([pxx, pxx2], axis=0)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="mic")

    f, pxx = spsig.welch(np.tile(signals["loudspeaker_array0"], (1,num_tile)), fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    f, pxx2 = spsig.welch(np.tile(signals["loudspeaker_array1"], (1,num_tile)), fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    pxx = np.concatenate([pxx, pxx2], axis=0)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="loudspeaker")

    f, pxx = spsig.welch(noise["mic_array"], fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="noise")

    ax.set_title("Signal spectrum of stationary microphones")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectrum [dB]")
    ax.legend()
    utils.set_basic_plot_look(ax)
    utils.save_plot("pdf", fig_folder, "spectrum_array")

    info, _, signals_fast, _  = ld.load(room_name, seq_len_ms, max_freq, speed="fast", downsampled=True, dataset_folder=dataset_path)
    _, _, noise_fast  = ld.load_noise(room_name, max_freq, speed="fast", downsampled=True, dataset_folder=dataset_path)

    # Plot energy levels for the moving microphone
    fig, ax = plt.subplots(1,1)
    f, pxx = spsig.welch(signals["mic_moving"], fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="mic slow")

    f, pxx = spsig.welch(signals_fast["mic_moving"], fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="mic fast")

    f, pxx = spsig.welch(signals["loudspeaker_moving"], fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="loudspeaker")

    f, pxx = spsig.welch(noise["mic_moving"], fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="noise slow")

    f, pxx = spsig.welch(noise_fast["mic_moving"], fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="noise fast")

    f, pxx = spsig.welch(noise["mic_moving_ambient"], fs = info["samplerate"], nperseg=fft_len, scaling="spectrum", axis=-1)
    ax.plot(f, 10 * np.log10(np.mean(pxx, axis=0)), label="noise ambient")

    ax.set_title("Signal spectrum for moving microphone")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectrum [dB]")
    ax.legend()
    utils.set_basic_plot_look(ax)
    utils.save_plot("pdf", fig_folder, "spectrum_moving_mic")

    # pass
    # kde = spstats.gaussian_kde(pxx[:,0])
    # top_factor = 3
    # bottom_factor = 1/3
    # num_points = 1000
    # density = kde(np.linspace(np.min(pxx[:,10]) * bottom_factor, top_factor * np.max(pxx[:,10]), num_points))


def plot_positions(dataset_path, fig_folder):
    info, pos, signals, rir  = ld.load("a", 1000, 4000, speed="fast", downsampled=True, dataset_folder=dataset_path)

    fig, ax = plt.subplots(1,1)

    ax.plot(pos["mic"][:,0], pos["mic"][:,1], marker="o", label="array")
    ax.plot(pos["loudspeaker"][:,0], pos["loudspeaker"][:,1], marker="s", label="loudspeaker")

    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    utils.set_basic_plot_look(ax)
    utils.save_plot("pdf", fig_folder, "positions")

def signal_spectograms(dataset_path, fig_folder):
    info, pos, signals, rir  = ld.load("a", 1000, 4000, speed="fast", downsampled=True, dataset_folder=dataset_path)
    _, pos_noise, noise  = ld.load_noise("a", 4000, speed="fast", downsampled=True, dataset_folder=dataset_path)
    stft_len = 256
    
    freqs_sig, t_sig, spec_sig = spsig.spectrogram(signals["mic_moving"][0,:], info["samplerate"], window="hann", nperseg=stft_len, nfft=stft_len)
    freqs_noise, t_noise, spec_noise = spsig.spectrogram(noise["mic_moving"][0,:], info["samplerate"], window="hann", nperseg=stft_len, nfft=stft_len)
    #freqs, t, spec2 = spsig.spectrogram(signals["mm_moving_noise_slow"][1,:], info["samplerate"], window="hann", nperseg=stft_len, nfft=stft_len)

    spec_sig = 10 * np.log10(spec_sig)
    spec_noise = 10 * np.log10(spec_noise)

    vmin = np.min([np.min(spec_sig), np.min(spec_noise)])
    vmax = np.max([np.max(spec_sig), np.max(spec_noise)])

    fig, axes = plt.subplots(2,1)
    
    clr = axes[0].imshow(spec_sig, interpolation="none", extent=[np.min(t_sig), np.max(t_sig), np.min(freqs_sig), np.max(freqs_sig)], origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(clr)
    cbar.set_label('Spectrum (dB)')

    clr = axes[1].imshow(spec_noise, interpolation="none", extent=[np.min(t_noise), np.max(t_noise), np.min(freqs_noise), np.max(freqs_noise)], origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(clr)
    cbar.set_label('Spectrum (dB)')
    # clr = axes[1].imshow(np.log10(spec2), interpolation="none", extent=[np.min(t), np.max(t), np.min(freqs), np.max(freqs)], origin='lower')
    # cbar = plt.colorbar(clr)
    # cbar.set_label('Spectrum')

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.axis("auto")
    
    utils.save_plot("pdf", fig_folder, "spectogram_moving_mic")



def main():
    dataset_path = pathlib.Path("c:/research/research_documents/202401_moving_mic_measurements/dataset")
    fig_folder = pathlib.Path(__file__).parent / "figs_analyze_dataset"
    fig_folder.mkdir(parents=True, exist_ok=True)

    estimate_rt60(dataset_path, fig_folder)
    plot_positions(dataset_path, fig_folder)
    
    signal_spectograms(dataset_path, fig_folder)
    plot_signal_levels(dataset_path, fig_folder)
    measure_speed_of_microphone(dataset_path, fig_folder)

if __name__ == "__main__":
    main()