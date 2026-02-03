import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import scipy.signal as spsig

import aspcol.utilities as utils
import aspcore.fouriertransform as ft
import aspcore.utilities as coreutils
import aspsim.diagnostics.diagnostics as dg
import aspsim.saveloadsession as sls

import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import load_dataset as ld


def load_exp_data(output_method="pdf", max_freq=500, room="a", dataset_folder=None):
    if dataset_folder is None:
        dataset_folder = pathlib.Path("c:/research/research_documents/202401_moving_mic_measurements/romms_dataset")
    info, pos, signals, rir = ld.load(room=room, seq_len_ms=500, max_freq=max_freq, speed = "fast", downsampled = True, dataset_folder=dataset_folder)
    _, pos_noise, noise = ld.load_noise(room=room, max_freq=max_freq, speed = "fast", downsampled = True, dataset_folder=dataset_folder)

    pos_moving, rev1_idxs, rev2_idxs = make_single_revolution_trajectory(pos["mic_moving"], info["seq_len"])
    sig_moving = np.concatenate((signals["mic_moving"][0,rev1_idxs[0]: rev1_idxs[1]], signals["mic_moving"][1,rev2_idxs[0]: rev2_idxs[1]]), axis=-1)
    loudspeaker_moving = np.concatenate((signals["loudspeaker_moving"][0,rev1_idxs[0]: rev1_idxs[1]], signals["loudspeaker_moving"][0,rev2_idxs[0]: rev2_idxs[1]]), axis=-1)
    noise_moving = np.concatenate((noise["mic_moving"][0,rev1_idxs[0]: rev1_idxs[1]], noise["mic_moving"][1,rev2_idxs[0]: rev2_idxs[1]]), axis=-1)
    signals["noise_moving"] = noise["mic_moving"]
    signals["noise_stationary"] = noise["mic_array"]

    num_periods = -1
    if num_periods > 0:
        sig_len_shortened = num_periods*info["seq_len"]
        pos_moving = pos_moving[:sig_len_shortened,:]
        sig_moving = sig_moving[:sig_len_shortened]
        loudspeaker_moving = loudspeaker_moving[:sig_len_shortened]
        noise_moving = noise_moving[:sig_len_shortened]
    num_periods = loudspeaker_moving.shape[0] // info["seq_len"]

    assert np.all([np.allclose(loudspeaker_moving[:info["seq_len"]], loudspeaker_moving[i*info["seq_len"]:(i+1)*info["seq_len"]]) for i in range(num_periods)]), "loudspeaker signal should be periodic"
    loudspeaker_moving = loudspeaker_moving[:info["seq_len"]]

    figure_folder = pathlib.Path(__file__).parent /"figs"# "c:/research/research_documents/202401_moving_mic_measurements/code_measure/figures")
    figure_folder = utils.get_unique_folder("figs_", figure_folder, detailed_naming=False)
    figure_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {figure_folder}")

    square_side = 0.6
    pos_eval_x = np.linspace(-square_side, square_side, 32)
    pos_eval_y = np.linspace(-square_side, square_side, 32)
    pos_eval = np.array(np.meshgrid(pos_eval_x, pos_eval_y)).T.reshape(-1, 2)
    pos_eval = np.concatenate([pos_eval, np.zeros((pos_eval.shape[0], 1))], axis=-1)
    
    fft_len = rir.shape[-1]
    freq_domain_rir = ft.rfft(rir)
    wave_num = ft.get_real_wavenum(fft_len, info["samplerate"], info["c"])
    freqs = ft.get_real_freqs(fft_len, info["samplerate"])

    speed = np.linalg.norm(np.diff(pos["mic_moving"][0,:,:], axis=0), axis=1) * info["samplerate"]
    median_speed = np.median(speed)
    info["speed_outer"] = median_speed

    plot_pos(pos_moving, pos, figure_folder, output_method=output_method, name_suffix="_real_data")
    plot_angle_deviation(info, pos_moving, figure_folder, output_method=output_method)

    noise_power = np.mean(noise["mic_moving"]**2, axis=-1)
    signal_plus_noise_power = np.mean(signals["mic_moving"]**2, axis=-1)
    snr = signal_plus_noise_power / noise_power - 1
    snr_db = 10 * np.log10(snr)
    with open(figure_folder / "snr_info_original_dataset.json", "w") as f:
        snr_info = {
            "noise_power" : noise_power.astype(float).tolist(),
            "signal_plus_noise_power" : signal_plus_noise_power.astype(float).tolist(),
            "snr" : snr.astype(float).tolist(),
            "snr_db" : snr_db.astype(float).tolist(),
            "mean_snr" : float(np.mean(signal_plus_noise_power) / np.mean(noise_power) - 1),
            "mean_snr_db" : float(10 * np.log10(np.mean(signal_plus_noise_power) / np.mean(noise_power) - 1))
        }
        json.dump(snr_info, f, indent=4)
    noise_power = np.mean(noise_moving**2, axis=-1)
    signal_plus_noise_power = np.mean(sig_moving**2, axis=-1)
    snr = signal_plus_noise_power / noise_power - 1
    snr_db = 10 * np.log10(snr)
    with open(figure_folder / "snr_info_data_used_in_experiment.json", "w") as f:
        snr_info = {
            "noise_power" : noise_power.astype(float).tolist(),
            "signal_plus_noise_power" : signal_plus_noise_power.astype(float).tolist(),
            "snr" : snr.astype(float).tolist(),
            "snr_db" : snr_db.astype(float).tolist(),
        }
        json.dump(snr_info, f, indent=4)



    return info, pos, signals, rir, pos_moving, sig_moving, loudspeaker_moving, noise_moving, pos_eval, freq_domain_rir, wave_num, freqs, figure_folder





def plot_angle_deviation(info, pos_moving, figure_folder, output_method="pdf"):
    angles = np.unwrap(utils.cart2spherical(pos_moving)[1][:,0])
    angles_dev = angles - np.arange(angles.shape[-1]) * np.mean(np.diff(angles))

    highpass_freqs = [0.1, 1, 10, 100]
    fig, axes = plt.subplots(len(highpass_freqs) + 1,1, figsize=(8, 2 * (1+len(highpass_freqs))))
    axes[0].plot(angles_dev, label="angles-deviation")

    angles_dev_filtered_power = {
        "no filter" : np.mean(angles_dev**2)
    }
    for i, highpass_freq in enumerate(highpass_freqs):
        sos = spsig.butter(1, highpass_freq, "highpass", fs=info["samplerate"], output="sos")
        angles_dev_filtered = spsig.sosfiltfilt(sos, angles_dev)
        axes[i+1].plot(angles_dev_filtered, label=f"angles-deviation HP:{highpass_freq} Hz")
        angles_dev_filtered_power[highpass_freq] = np.mean(angles_dev_filtered**2)
    for ax in axes:
        ax.legend()
        ax.set_title("Azimuth angles")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Azimuth angle (rad)")
        coreutils.set_basic_plot_look(ax)
    coreutils.save_plot(output_method, figure_folder, "angles_deviation_real")

    with open(figure_folder / "angles_deviation_power.json", "w") as f:
        json.dump(angles_dev_filtered_power, f, indent=4)




def make_double_revolution_trajectory(pos, seq_len):
    """Takes positions of two moving microphones and merges them into a single trajectory

    Assumes that the microphones have been moved two revolutions around a circle. 
    
    Parameters
    ----------
    pos : np.ndarray of shape (num_mics, num_samples, 3)
        position of the moving microphones that should be merged into a single trajectory

    Returns
    -------
    pos_traj : np.ndarray of shape (num_traj, 3)
        the merged trajectory, the exact number of samples will vary
    first_rev_idxs : tuple of two ints
        the start and end indices of the first revolution
    second_rev_idxs : tuple of two ints
        the start and end indices of the second revolution
    """
    assert pos.ndim == 3
    assert pos.shape[0] == 2, "only two microphones are supported"
    assert pos.shape[2] == 3, "only 3D positions are supported"
    num_samples = pos.shape[1]
    num_periods = num_samples / seq_len

    angles = utils.cart2pol(pos[0,:,0], pos[0,:,1])[1]

    start_idx = num_samples // 4
    start_angle = angles[start_idx]
    one_revolution_candidate_indices = np.logical_and(np.abs(angles - start_angle) < 1e-2, np.concatenate((np.zeros(num_samples//2) , np.ones(num_samples - num_samples//2))))
    one_revolution_idx = int(np.mean(np.arange(num_samples)[one_revolution_candidate_indices])) #exact value is not important, just take mean of candidates

    # Find end point of first revolution with an integer number of periods close to exactly one revolution
    candidate_end_idx1 = start_idx + seq_len * ((one_revolution_idx - start_idx) // seq_len)
    candidate_end_idx2 = candidate_end_idx1 + seq_len
    end_idx = candidate_end_idx1 if np.abs(candidate_end_idx1 - one_revolution_idx) < np.abs(candidate_end_idx2 - one_revolution_idx) else candidate_end_idx2
    end_angle = angles[end_idx]

    #Choose start idx for second revolution that is as close as possible to where the first revolution ended
    second_rev_candidate_indices = np.logical_and(np.abs(angles - end_angle) < 1e-2, np.concatenate((np.ones(num_samples//2) , np.zeros(num_samples - num_samples//2))))
    second_rev_idx = int(np.mean(np.arange(num_samples)[second_rev_candidate_indices]))
    second_rev_idx1 = start_idx + seq_len * ((second_rev_idx - start_idx) // seq_len)
    second_rev_idx2 = second_rev_idx1 + seq_len
    angle_diff1 = np.abs((angles[second_rev_idx1] - end_angle + np.pi) % (2 * np.pi) - np.pi)
    angle_diff2 = np.abs((angles[second_rev_idx2] - end_angle + np.pi) % (2 * np.pi) - np.pi)
    start_idx_second_rev = second_rev_idx1 if angle_diff1 < angle_diff2 else second_rev_idx2

    assert (end_idx - start_idx) % seq_len == 0, "first revolution should have an integer number of periods"
    assert (end_idx - start_idx_second_rev) % seq_len == 0, "second revolution should have an integer number of periods"

    pos_traj = np.concatenate((pos[0,start_idx:end_idx,:], pos[1,start_idx_second_rev:end_idx,:]), axis=0)
    return pos_traj, (start_idx, end_idx), (start_idx_second_rev, end_idx)


def make_single_revolution_trajectory(pos, seq_len):
    """Takes positions of two moving microphones and merges them into a single trajectory

    Assumes that the microphones have been moved two revolutions around a circle. 
    
    Parameters
    ----------
    pos : np.ndarray of shape (num_mics, num_samples, 3)
        position of the moving microphones that should be merged into a single trajectory

    Returns
    -------
    pos_traj : np.ndarray of shape (num_traj, 3)
        the merged trajectory, the exact number of samples will vary
    first_rev_idxs : tuple of two ints
        the start and end indices of the first revolution
    second_rev_idxs : tuple of two ints
        the start and end indices of the second revolution
    """
    assert pos.ndim == 3
    assert pos.shape[0] == 2, "only two microphones are supported"
    assert pos.shape[2] == 3, "only 3D positions are supported"
    num_samples = pos.shape[1]
    num_periods = num_samples / seq_len

    angles = utils.cart2pol(pos[0,:,0], pos[0,:,1])[1]

    start_idx = num_samples // 8
    start_angle = angles[start_idx]
    half_revolution_candidate_indices = np.logical_and(np.abs(angles - start_angle - np.pi) < 1e-2, np.concatenate((np.zeros(num_samples//4), np.ones(num_samples//2), np.zeros(num_samples - num_samples//2 - num_samples//4))))
    half_revolution_idx = int(np.mean(np.arange(num_samples)[half_revolution_candidate_indices])) #exact value is not important, just take mean of candidates

    # Find end point of first revolution with an integer number of periods close to exactly one revolution
    candidate_end_idx1 = start_idx + seq_len * ((half_revolution_idx - start_idx) // seq_len)
    candidate_end_idx2 = candidate_end_idx1 + seq_len
    end_idx = candidate_end_idx1 if np.abs(candidate_end_idx1 - half_revolution_idx) < np.abs(candidate_end_idx2 - half_revolution_idx) else candidate_end_idx2
    end_angle = angles[end_idx]

    #Choose start idx for second revolution that is as close as possible to where the first revolution ended
    start_idx_second_mic = end_idx
    start_angle = angles[start_idx_second_mic]
    half_revolution_candidate_indices = np.logical_and(np.abs(angles - start_angle + np.pi) < 1e-2, np.concatenate((np.zeros(num_samples//2), np.ones(num_samples - num_samples//2))))
    half_revolution_idx = int(np.mean(np.arange(num_samples)[half_revolution_candidate_indices])) #exact value is not important, just take mean of candidates

    # Find end point of first revolution with an integer number of periods close to exactly one revolution
    candidate_end_idx1 = start_idx_second_mic + seq_len * ((half_revolution_idx - start_idx_second_mic) // seq_len)
    candidate_end_idx2 = candidate_end_idx1 + seq_len
    end_idx_second_mic = candidate_end_idx1 if np.abs(candidate_end_idx1 - half_revolution_idx) < np.abs(candidate_end_idx2 - half_revolution_idx) else candidate_end_idx2

    assert (end_idx - start_idx) % seq_len == 0, "first revolution should have an integer number of periods"
    assert (end_idx - start_idx_second_mic) % seq_len == 0, "second revolution should have an integer number of periods"

    pos_traj = np.concatenate((pos[0,start_idx:end_idx,:], pos[1,start_idx_second_mic:end_idx_second_mic,:]), axis=0)
    return pos_traj, (start_idx, end_idx), (start_idx_second_mic, end_idx_second_mic)






def plot_pos(pos_moving, pos, figure_folder, output_method="pdf", name_suffix=""):
    fig, ax = plt.subplots(1,1, figsize=(10,10))

    ax.plot(pos_moving[:,0], pos_moving[:,1], marker="x", label="moving mic", linestyle="None")
    jump_idx = _find_jump_idx(pos_moving)
    ax.plot([pos_moving[jump_idx,0], pos_moving[jump_idx+1,0]], [pos_moving[jump_idx,1], pos_moving[jump_idx+1,1]], label="jump")

    ax.plot(pos_moving[0,0], pos_moving[0,1], marker="o", label="start", linestyle="None")
    ax.plot(pos_moving[-1,0], pos_moving[-1,1], marker="s", label="end", linestyle="None")
    ax.plot(pos["mic"][:,0], pos["mic"][:,1], label="microphones", marker="o", linestyle="None")
    ax.plot(pos["loudspeaker"][:,0], pos["loudspeaker"][:,1], label="loudspeakers", marker="s", linestyle="None")
    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis('equal')
    coreutils.set_basic_plot_look(ax)
    coreutils.save_plot(output_method, figure_folder, f"positions{name_suffix}")

def _find_jump_idx(pos_moving):
    traj_diff = np.linalg.norm(np.diff(pos_moving, n = 1, axis=0), axis=1)
    median_traj_diff = np.median(traj_diff)
    jump_idx = np.argmax(traj_diff)

    if traj_diff[jump_idx] < 100 * median_traj_diff:
        print(f"Likely that jump index is incorrect, median diff: {median_traj_diff}, jump diff: {traj_diff[jump_idx]}")
    return jump_idx




# Functions for simulation experiments are below

def run_and_save(sim):
    sim.diag.add_diagnostic("mic", dg.RecordSignal("mic", sim.sim_info, num_channels=sim.arrays["mic"].num, export_func="npz"))
    sim.diag.add_diagnostic("eval", dg.RecordSignal("eval", sim.sim_info, num_channels=sim.arrays["eval"].num, export_func="npz"))
    sim.diag.add_diagnostic("mic_dynamic", dg.RecordSignal("mic_dynamic", sim.sim_info, num_channels=sim.arrays["mic_dynamic"].num, export_func="npz"))
    sim.diag.add_diagnostic("audio_mic", dg.RecordSignal("mic", sim.sim_info, num_channels=1, channel_idx = 3, export_func="wav"))
    sim.diag.add_diagnostic("audio_mic_dynamic", dg.RecordSignal("mic_dynamic", sim.sim_info, num_channels=1, channel_idx=0, export_func="wav"))
    sim.diag.add_diagnostic("src", dg.RecordSignal("src", sim.sim_info, num_channels=sim.arrays["src"].num, export_func="npz"))

    if "image" in sim.arrays:
        sim.diag.add_diagnostic("image", dg.RecordSignal("image", sim.sim_info, num_channels=sim.arrays["image"].num, export_func="npz"))
    
    sim.run_simulation()
    sim.arrays.save_to_file(sim.folder_path)
    sim.sim_info.save_to_file(sim.folder_path)

def load_session(fig_folder):
    with open(fig_folder.joinpath("extra_parameters.json")) as f:
        extra_params = json.load(f)
    signal_paths = get_signal_paths(fig_folder)
    sig = load_npz(signal_paths)
    sim_info, arrays = sls.load_from_path(fig_folder)    
    seq_len = arrays["src"].source.tot_samples
    pos_dyn = arrays["mic_dynamic"].pos[:,None,:]
    sig["mic_dynamic"] = load_dynamic_sig(sig["mic_dynamic"], seq_len)

    extra_params["pseq_start_idx"] = sim_info.sim_buffer + extra_params["initial_delay"]
    return sig, sim_info, arrays, pos_dyn, seq_len, extra_params

def load_dynamic_sig(sig_raw, seq_len):
    """
    takes the many statoinary microphones and puts it together into one seemingly 
    moving microphone. 
    """
    sig_raw = sig_raw[:,:seq_len]
    sig_len = sig_raw.shape[0]

    assert sig_len % seq_len == 0
    num_periods = sig_len // seq_len
    sig_out = np.zeros((1, sig_len))
    for i in range(sig_len):
        sig_out[0,i] = sig_raw[i,i % seq_len]
    return sig_out

def load_npz(signal_paths):
    sig = {}
    for sig_name, sig_path in signal_paths.items():
        loaded_data = np.load(sig_path)
        dict_data = {key: data for key, data in loaded_data.items()}
        for key, data in dict_data.items():
            assert key not in sig
            sig[key] = data
    return sig

def get_signal_paths(fig_folder):
    signal_paths = {}
    for f in fig_folder.iterdir():
        if f.suffix == ".npz":
            sig_name_components = f.stem.split("_")
            sig_name = "_".join(sig_name_components[:-1])
            signal_paths[sig_name] = f
    return signal_paths

