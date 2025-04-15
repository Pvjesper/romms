import numpy as np
import pathlib
import json
import yaml
import copy
import scipy.signal as spsig
import aspsim.room.generatepoints as gp
import aspsim.room.region as reg
from aspsim.simulator import SimulatorSetup
import aspsim.signal.sources as sources

import load_dataset as ld
import matplotlib.pyplot as plt

import aspcore.pseq as pseq
import aspcore.filterdesign as fd
import aspcore.fouriertransform as ft
import aspcore.utilities as asputil

import aspcol.soundfieldestimation as sfe
import aspcol.utilities as utils
import aspcol.plot as aspplot
import latexutilities.pgfplotutilities as latexutil

import exp_funcs_ideal_sampling as exis
import exp_funcs_extra

def run_single_exp(fig_folder, snrs=[np.inf], pos_noise_power=[0], noise_type="white", pos_noise_type="gaussian", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(snrs, (list, tuple, np.ndarray)):
        snrs = [snrs]
    if not isinstance(pos_noise_power, (list, tuple, np.ndarray)):
        pos_noise_power = [pos_noise_power]

    sig, sim_info, arrays, pos_dyn, seq_len, extra_params = exis.load_session(fig_folder)

    #print(f"Simulation start position: {pos_dyn[0,0,:]}, in polar coords: {gp.cart2pol(pos_dyn[0,0,0], pos_dyn[0,0,1])}")
    #print(f"Start position after init delay: {pos_dyn[extra_params['initial_delay'],0,:]}, in polar coords: {gp.cart2pol(pos_dyn[extra_params['initial_delay'],0,0], pos_dyn[extra_params['initial_delay'],0,1])}")

    #p = sig["mic"][:,-seq_len:]
    #p_eval = sig["eval"][:,-seq_len:]
    if pos_dyn.shape[1] != 1:
        raise NotImplementedError
    pos_dyn = pos_dyn[:,0,:]
    exp_funcs_extra.plot_pos(pos_dyn, {"mic" : arrays["mic"].pos, "eval" : arrays["eval"].pos, "loudspeaker" : arrays["src"].pos}, fig_folder, OUTPUT_METHOD, "_used_in_sim")

    plot_pos_for_paper(pos_dyn, arrays["mic"].pos, arrays["src"].pos, sim_info.samplerate, fig_folder, output_method=OUTPUT_METHOD, name_suffix="_sim_data_for_paper")


    # ====== ADD POSITION NOISE AND SIGNAL NOISE======
    pos_dyn_noisy = add_noise_to_pos_dyn(pos_dyn, pos_noise_power, noise_type=pos_noise_type, rng=rng, fig_folder=fig_folder)
    p_dyn_noisy, noise_only = add_noise_to_p_dyn(sig["mic_dynamic"], snrs, noise_type, rng, fig_folder)

    p_stationary = sig["mic"][:,-seq_len:]
    p_stationary_noisy, noise_only_stationary = add_noise_to_p_stationary(p_stationary, snrs, noise_type, rng, fig_folder)
    pos_stat_noisy = add_noise_to_pos_dyn(arrays["mic"].pos, pos_noise_power, noise_type=pos_noise_type, rng=rng, fig_folder=fig_folder)

    #rir_mic_noisy = {snr_val : pseq.decorrelate(p, sig["src"][:,-seq_len:]) for snr_val, p in p_stationary_noisy.items()}
    #rir_mic_no_noise = pseq.decorrelate(p_stationary, sig["src"][:,-seq_len:])
    #print(f"MSE between pseq rir and true rir: {np.mean((arrays.paths['src']['mic'] - rir_mic_no_noise)**2)}")
    #rir_mic_freq_noisy = {snr_val : ft.rfft(rir_mic) for snr_val, rir_mic in rir_mic_noisy.items()}
    #rir_mic_freq_no_noise = ft.rfft(rir_mic_no_noise)

    sequence = sig["src"][0,extra_params["initial_delay"]:]

    rir_eval = arrays.paths["src"]["eval"]
    rir_eval_freq = ft.rfft(rir_eval)

    #rir_mic = arrays.paths["src"]["mic"]
    #rir_mic_freq = ft.rfft(rir_mic)
    
    #freqs = np.arange(seq_len) * sim_info.samplerate / seq_len
    #k = 2 * np.pi * freqs / sim_info.c
    center = np.array(extra_params["center"])[None,:]
    
    real_freqs = ft.get_real_freqs(seq_len, sim_info.samplerate)

    lambda_inv = 1e-1
    pos_noise_power_factor = 10
    print(f"lambda_inv: {lambda_inv}")
    print(f"pos_noise_power_factor: {pos_noise_power_factor}")
    noise_powers = {snr_val : np.mean(noise_only[snr_val]**2) for snr_val in snrs}
    reg_param_td_snr = {snr_val : np.max((noise_powers[snr_val] * lambda_inv, 1e-8)) for snr_val in snrs}
    reg_param_td_pos = {pnp : np.max((pos_noise_power_factor * pnp * lambda_inv, 1e-8)) for pnp in pos_noise_power}

    reg_param_td = {}
    for snr_val, rp1 in reg_param_td_snr.items():
        reg_param_td[snr_val] = {}
        for pnp, rp2 in reg_param_td_pos.items():
            reg_param_td[snr_val][pnp] = np.max((rp1, rp2))

    #reg_param_td = {snr_val : np.max((noise_powers[snr_val], pnp, 1e-8)) for snr_val in snrs for pnp in pos_noise_power}

    reg_param = 1e-4
    spatial_spectrum_reg = 1e-3
    print(f"spatial_spectrum_reg: {spatial_spectrum_reg}")
    #reg_param_td = 1e-4
    
    estimates = {}
    diag = {}

    for (snr_val, p_dyn), (snr_val_stat, p_stat) in zip(p_dyn_noisy.items(), p_stationary_noisy.items()):
        assert np.allclose(snr_val, snr_val_stat)
        for (pd_pow, pd), (pd_pow_stat, pd_stat) in zip(pos_dyn_noisy.items(), pos_stat_noisy.items()):
            assert np.allclose(pd_pow, pd_pow_stat)

            ss_reg = np.max((reg_param_td[snr_val][pd_pow], spatial_spectrum_reg))
            #ss_reg = reg_param_td[snr_val][pd_pow]

            print(f"Calculating moving omni for pos noise power {pd_pow} and snr {snr_val}")
            estimates[f"moving omni snr:{snr_val} pos_noise:{pd_pow}"] = sfe.inf_dimensional_shd_dynamic(p_dyn, pd, arrays["eval"].pos, sequence[:seq_len], sim_info.samplerate, sim_info.c, reg_param_td[snr_val][pd_pow])
            estimates[f"spatial spectrum snr:{snr_val} pos_noise:{pd_pow}"] = sfe.est_spatial_spectrum_dynamic(p_dyn, pd, arrays["eval"].pos, sequence[:seq_len], sim_info.samplerate, sim_info.c, ss_reg, verbose=False)

            estimates[f"kernel interpolation snr:{snr_val} pos_noise:{pd_pow}"] = sfe.est_ki_diffuse(p_stat, sig["src"][:,-seq_len:], pd_stat, arrays["eval"].pos, sim_info.samplerate, sim_info.c, reg_param_td[snr_val][pd_pow])

    estimates[f"noise free moving omni"] = sfe.inf_dimensional_shd_dynamic(p_dyn, pos_dyn, arrays["eval"].pos, sequence[:seq_len], sim_info.samplerate, sim_info.c, 1e-8)
    estimates[f"noise free spatial spectrum"] = sfe.est_spatial_spectrum_dynamic(p_dyn, pd, arrays["eval"].pos, sequence[:seq_len], sim_info.samplerate, sim_info.c, spatial_spectrum_reg, verbose=False)
    estimates[f"noise free kernel interpolation"] = sfe.est_ki_diffuse(p_stationary, sig["src"][:,-seq_len:], arrays["mic"].pos, arrays["eval"].pos, sim_info.samplerate, sim_info.c, 1e-8)


    #estimates["nearest neighbour"] = sfe.pseq_nearest_neighbour(p, sequence[:seq_len], arrays["mic"].pos, arrays["eval"].pos)
    #estimates["kernel interpolation"] = sfe.est_ki_diffuse(p, sequence[:seq_len], arrays["mic"].pos, arrays["eval"].pos, sim_info.samplerate, sim_info.c, reg_param)
  
    #r_max = np.linalg.norm(pos_dyn, axis=-1) #0.6
    # estimates["katzberg"], diag["katzberg"] = sfe.est_spatial_spectrum_dynamic(p_dyn, pos_dyn, arrays["eval"].pos, sequence[:seq_len], sim_info.samplerate, sim_info.c, r_max, verbose=True)
    #diag["katzberg"]["r_max"] = r_max

    with open(fig_folder / "reg_params_td.json", "w") as f:
        json.dump(reg_param_td, f, indent=4)

    #estimates["1D-interpolation"] = hse.est_rir_circle(p_dyn, pos_dyn, arrays["eval"].pos, sequence[:seq_len], sim_info.samplerate, center, int_order=512)

    with open(fig_folder.joinpath("diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=4)

    estimates_all = {key : val for key, val in estimates.items()}
    estimates_all["true"] = rir_eval_freq
    np.savez(fig_folder.joinpath("estimates.npz"), **estimates_all)


    aspplot.soundfield_estimation_comparison(arrays["eval"].pos, estimates, rir_eval_freq, real_freqs, fig_folder, shape="rectangle", center=center, output_method=OUTPUT_METHOD, pos_mic=arrays["mic"].pos, num_examples = 10, remove_freqs_below=50)
    #if num_ls == 1:
    #    test_pseq.run_from_data(p, sequence, seq_len, sim_info.samplerate, extra_params["max_sweep_freq"],arrays, fig_folder, sim_info, extra_params)




def add_noise_to_pos_dyn(pos_dyn, noise_power=1e-4, noise_type="gaussian", rng=None, fig_folder = None):
    """
    Parameters
    ----------
    pos_dyn : np.ndarray of shape (num_samples, 3)
        The dynamic position data.
    noise_power : float or list of floats, optional
        The power of the noise to add. 
        If a list is given, multiple versions of the noisy position data will be returned
    
    Returns
    -------
    pos_noise : dict of np.ndarray with same shape as pos_dyn
        The noisy position data. The keys are the noise powers.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(noise_power, (list, tuple, np.ndarray)):
        noise_power = [noise_power]
    noise_power = np.array(noise_power)
    assert noise_power.ndim == 1

    pos_noisy = {}
    if noise_type == "gaussian":
        pos_noise_prototype = rng.normal(loc=0, scale=1, size=pos_dyn.shape)
        for npow in noise_power:
            pos_noisy[npow] = pos_dyn + np.sqrt(npow) * pos_noise_prototype
            #print (f"Added noise with power {npow} to position data")
            #print(f"Has an empirical variance of {np.mean((pos_noise[npow] - pos_dyn)**2)}")
    elif noise_type == "angles":
        r, angles = utils.cart2spherical(pos_dyn)
        azimuth_noise = rng.normal(loc=0, scale=1, size=angles.shape[0])
        for npow in noise_power:
            angles_noisy = np.copy(angles)
            angles_noisy[:,0] += np.sqrt(npow) * azimuth_noise
            pos_noisy[npow] = utils.spherical2cart(r, angles_noisy)
    else:
        raise ValueError("Unknown noise type")
    
    if fig_folder is not None:
        empirical_variance = {npow.item() : np.mean((pos_noisy[npow] - pos_dyn)**2) for npow in noise_power}
        with open(fig_folder / "mean_square_position_error.json", "w") as f:
            json.dump(empirical_variance, f, indent=4)

    if fig_folder is not None:
        fig, axes = plt.subplots(1,1+len(noise_power), figsize=(4 + 4 * len(noise_power),6))
        angles = utils.cart2spherical(pos_dyn)[1]
        axes[0].plot(angles[:,0], label="original")
        for i, npow in enumerate(noise_power):
            angles_noisy = utils.cart2spherical(pos_noisy[npow])[1]
            axes[i].plot(angles_noisy[:,0], label=f"noisy {npow}")
        for ax in axes:
            ax.legend()
            ax.set_title("Azimuth angles")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Azimuth angle (rad)")
            aspplot.set_basic_plot_look(ax)
        aspplot.save_plot(OUTPUT_METHOD, fig_folder, "angles")
    return pos_noisy
    
def add_noise_to_p_dyn(p_dyn, snr, noise_type="white", rng=None, fig_folder=None):
    if rng is None:
        rng = np.random.default_rng()
    if noise_type == "white":
        noise_sig = rng.normal(0, 1, p_dyn.shape)
    elif noise_type == "real":
        assert fig_folder is not None
        noise_sig = np.load(fig_folder / "noise_moving.npy")

    p_dyn_noisy = {}
    noise_only = {}

    for snr_val in snr:
        signal_power = np.mean(p_dyn**2)
        noise_power = np.mean(noise_sig**2)
        current_snr = signal_power / noise_power
        if np.isinf(snr_val):
            noise_pow_factor = 0
        else:
            noise_pow_factor = current_snr / utils.db2pow(snr_val)

        noise_only[snr_val] = np.sqrt(noise_pow_factor) * noise_sig
        p_dyn_noisy[snr_val] = p_dyn + noise_only[snr_val]

    if fig_folder is not None:
        empirical_variance = {snr_val: np.mean((p_dyn_noisy[snr_val] - p_dyn)**2) for snr_val in snr}
        with open(fig_folder / "mean_square_signal_error.json", "w") as f:
            json.dump(empirical_variance, f, indent=4)

        empirical_snr = {snr_val: 10 * np.log10(np.mean(p_dyn**2) / np.mean(noise_only[snr_val]**2)) for snr_val in snr}
        with open(fig_folder / "empirical_snr.json", "w") as f:
            json.dump(empirical_snr, f, indent=4)

    #if np.inf not in snr:
    #    p_dyn_noisy[np.inf] = p_dyn
    return p_dyn_noisy, noise_only


def add_noise_to_p_stationary(p_stationary, snr, noise_type="white", rng=None, fig_folder=None):
    if rng is None:
        rng = np.random.default_rng()
    if noise_type == "white":
        noise_sig = rng.normal(0, 1, p_stationary.shape)
    elif noise_type == "real":
        assert fig_folder is not None
        noise_sig = np.load(fig_folder / "noise_moving.npy")

    assert np.prod(p_stationary.shape) <= np.prod(noise_sig.shape)
    if noise_sig.ndim == 1:
        noise_sig = noise_sig.reshape(-1, p_stationary.shape[-1])
        if noise_sig.shape[0] > p_stationary.shape[0]:
            noise_sig = noise_sig[:p_stationary.shape[0],:]
    else:
        raise NotImplementedError
    assert noise_sig.shape == p_stationary.shape

    p_stationary_noisy = {}
    noise_only = {}

    for snr_val in snr:
        signal_power = np.mean(p_stationary**2)
        noise_power = np.mean(noise_sig**2)
        current_snr = signal_power / noise_power
        if np.isinf(snr_val):
            noise_pow_factor = 0
        else:
            noise_pow_factor = current_snr / utils.db2pow(snr_val)

        noise_only[snr_val] = np.sqrt(noise_pow_factor) * noise_sig
        p_stationary_noisy[snr_val] = p_stationary + noise_only[snr_val]

    if fig_folder is not None:
        empirical_variance = {snr_val : np.mean((p_stationary_noisy[snr_val] - p_stationary)**2) for snr_val in snr}
        with open(fig_folder / "mean_square_signal_error.json", "w") as f:
            json.dump(empirical_variance, f, indent=4)

    #if np.inf not in snr:
    #    p_dyn_noisy[np.inf] = p_dyn
    return p_stationary_noisy, noise_only




def make_stationary_circle_pos(num_mics, radii, num_angles):
    num_angles = 8
    angles = np.arange(num_angles) * np.pi / 4
    (x1, y1) = utils.pol2cart(radii[0], angles)
    (x2, y2) = utils.pol2cart(radii[1], angles)
    x = []
    y = []
    for i in range(num_angles):
        x.append(x1[i])
        x.append(x2[i])
        y.append(y1[i])
        y.append(y2[i])
    x = np.array(x)
    y = np.array(y)
    pos_mic = np.stack((x, y, np.zeros_like(x)), axis=-1)
    return pos_mic



def filter_rirs(rir, sr, cutoff):
    sos = spsig.butter(4, cutoff, 'highpass', fs=sr, output='sos')

    pad = sr
    rir_padded = np.concatenate((np.zeros((*rir.shape[:-1], pad)), rir, np.zeros((*rir.shape[:-1], pad))), axis=-1)
    filtered_rir = spsig.sosfiltfilt(sos, rir_padded, axis=-1)
    filtered_rir = filtered_rir[...,pad:-pad]
    return filtered_rir

def get_noise_data(noise_type, num_noise, num_samples, samplerate, rng, real_noise):
    if noise_type == "white":
        noise_data = np.stack([rng.normal(0, 1, num_samples) for i in range(num_noise)], axis=0)
    elif noise_type == "real":
        noise_data = real_noise
    elif noise_type == "pink":    
        noise_data = np.stack([pink_noise(num_samples, samplerate, rng) for i in range(num_noise)], axis=0)
    else:
        raise ValueError("Unknown noise type")
    
    sos = spsig.butter(4, 10, 'highpass', fs=samplerate, output='sos')
    #rir_padded = np.concatenate((np.zeros((*rir.shape[:-1], pad)), rir, np.zeros((*rir.shape[:-1], pad))), axis=-1)
    noise_data = spsig.sosfiltfilt(sos, noise_data, axis=-1)

    return noise_data


def _circle_trajectory(radii, speed_outer, samplerate):
    radius_outer = np.max(radii)
    circumference_outer = 2 * np.pi * radius_outer
    time_outer = circumference_outer / speed_outer
    num_samples_single_rev = int(time_outer * samplerate)
    num_samples = 2 * num_samples_single_rev

    azimuth = -np.linspace(0, 4*np.pi, num_samples)
    zenith = np.ones(num_samples) * np.pi / 2
    angles = np.stack((azimuth, zenith), axis=-1)
    #r = np.ones(num_samples) * radii[]
    pos_traj = np.stack([utils.spherical2cart(r * np.ones(num_samples), angles) for r in radii], axis=0)

    return pos_traj


def generate_signals_2d():
    info, pos, signals, rir, pos_moving_real, sig_moving, loudspeaker_moving, noise_moving, pos_eval, freq_domain_rir, wave_num, freqs, figure_folder = exp_funcs_extra.load_exp_data(output_method=OUTPUT_METHOD)

    #
    #mean_speed = np.median(speed)
    radii = [0.5, 0.45]
    pos_moving_sim = _circle_trajectory(radii, info["speed_outer"], info["samplerate"])
    pos_moving_sim, rev1_idxs, rev2_idxs = exp_funcs_extra.make_single_revolution_trajectory(pos_moving_sim, info["seq_len"])
    #pos_moving_sim = pos_moving_sim[:pos_moving_real.shape[0],:]

    noise_moving = np.concatenate((signals["noise_moving"][0,rev1_idxs[0]: rev1_idxs[1]], signals["noise_moving"][1,rev2_idxs[0]: rev2_idxs[1]]), axis=-1)
    noise_moving = noise_moving[:pos_moving_sim.shape[0]]

    #pos_noise = np.array([[3, 1, 0]])
    highpass_cutoff = 50
    #num_samples_set_noise_power = info["seq_len"] * 10
    #rng = np.random.default_rng(123456)

    setup = SimulatorSetup(figure_folder)

    setup.sim_info.samplerate = info["samplerate"]
    sr = setup.sim_info.samplerate

    center = np.zeros(3)
    #pos_traj = pos_moving
    pos_src = pos["loudspeaker"]

    #pos_eval = pos["mic"]
    x_extent = (np.min(pos["mic"][:,0]), np.max(pos["mic"][:,0]))
    y_extent = (np.min(pos["mic"][:,1]), np.max(pos["mic"][:,1]))
    side_len = [x_extent[1] - x_extent[0], y_extent[1] - y_extent[0]]
    density = 0.05
    eval_reg = reg.Rectangle(side_len, (0,0,0), (density, density))
    pos_eval = eval_reg.equally_spaced_points()


    num_stationary_mics = pos_moving_sim.shape[0] // info["seq_len"]
    idx_offset = info["seq_len"] // 2
    mic_idxs = np.arange(num_stationary_mics) * info["seq_len"] + idx_offset
    pos_mic = pos_moving_sim[mic_idxs,:]

    #mic_idxs = np.array([0, 3, 31, 34, 25, 28, 56, 59, 10, 49])
    #mic_idxs = np.array([0, 31, 34, 25, 28, 59, 10, 49])
    #pos_mic = pos["mic"][mic_idxs,:]


    seq_len = info["seq_len"]
    initial_delay = sr #2*info["seq_len"]
    post_delay = 0

    setup.sim_info.tot_samples = initial_delay + seq_len
    setup.sim_info.sim_buffer = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 3.2, 2.5]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 = 0.22
    setup.sim_info.max_room_ir_length = seq_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 32
    setup.sim_info.plot_output = "pdf"

    #seq_len = setup.sim_info.max_room_ir_length
    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(sequence)

    # if noise_type == "white" or noise_type == "pink":
    #     noise_signal = get_noise_data(noise_type, pos_noise.shape[0], setup.sim_info.tot_samples, sr, rng)
    # noise_source = sources.Sequence(noise_signal)
    # setup.add_free_source("noise", pos_noise, noise_source)
    # setup.arrays.path_type["noise"]["eval"] = "none"

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_free_source("src", pos_src, sequence_src)
    setup.add_mics("mic_dynamic", pos_moving_sim)

    sim = setup.create_simulator()
    sim.arrays.paths["src"]["mic"] = filter_rirs(sim.arrays.paths["src"]["mic"], sr, highpass_cutoff)
    sim.arrays.paths["src"]["mic_dynamic"] = filter_rirs(sim.arrays.paths["src"]["mic_dynamic"], sr, highpass_cutoff)
    sim.arrays.paths["src"]["eval"] = filter_rirs(sim.arrays.paths["src"]["eval"], sr, highpass_cutoff)

    sim_rir_power_mean = np.mean(sim.arrays.paths["src"]["mic"]**2)
    DATASET_RIR_POWER_MEAN = 0.005438353038986198
    rir_power_factor = np.sqrt(DATASET_RIR_POWER_MEAN / sim_rir_power_mean)
    sim.arrays.paths["src"]["mic"] *= rir_power_factor
    sim.arrays.paths["src"]["mic_dynamic"] *= rir_power_factor
    sim.arrays.paths["src"]["eval"] *= rir_power_factor

    print(f"RIR power mean {np.mean(sim.arrays.paths['src']['mic']**2)}")
    print(f"RIR power mean {np.mean(sim.arrays.paths['src']['mic_dynamic']**2)}")
    #sim.arrays.paths["noise"]["mic"] = filter_rirs(sim.arrays.paths["noise"]["mic"], sr, highpass_cutoff)
    #sim.arrays.paths["noise"]["mic_dynamic"] = filter_rirs(sim.arrays.paths["noise"]["mic_dynamic"], sr, highpass_cutoff)

    speed = np.linalg.norm(np.diff(pos_moving_sim, axis=0), axis=1) * sr
    speed_real = np.linalg.norm(np.diff(pos_moving_real, axis=0), axis=1) * sr

    exis.run_and_save(sim)

    with open(sim.folder_path.joinpath("extra_parameters.json"), "w") as f:
        json.dump({"seq_len" : seq_len, 
                    "initial_delay" : initial_delay,
                "post_delay" : post_delay,
                "max_sweep_freq" : sr // 2,
                    "center" : center.tolist(), 
                    "downsampling_factor" : 1,
                    "speed min" : np.min(speed),
                    "speed max" : np.max(speed),
                    "speed mean" : np.mean(speed),
                    "speed median" : np.median(speed),
                    "total time" : pos_moving_sim.shape[0] / sr,
                    } ,f)
    
    np.save(sim.folder_path.joinpath("noise_moving.npy"), noise_moving)
    #np.save(sim.folder_path.joinpath("noise_array.npy"), noise_array)

    fig, axes = plt.subplots(2,1, figsize=(6,6))
    angles = utils.cart2spherical(pos_moving_real)[1]
    axes[0].plot(angles[:,0], label="real data")
    angles_noisy = utils.cart2spherical(pos_moving_sim)[1]
    axes[1].plot(angles_noisy[:,0], label=f"sim data")
    for ax in axes:
        ax.legend()
        ax.set_title("Azimuth angles")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Azimuth angle (rad)")
        aspplot.set_basic_plot_look(ax)
    aspplot.save_plot(OUTPUT_METHOD, sim.folder_path, "angles_sim_vs_real")

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.plot(speed, label="sim data")
    ax.plot(speed_real, label="real data")
    ax.set_title("Speed")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Speed (m/s)")
    aspplot.set_basic_plot_look(ax)
    aspplot.save_plot(OUTPUT_METHOD, sim.folder_path, "speed_sim_vs_real")

    #snr_folders = {}
    #for snr in snr_list:
    #    sim = setup.create_simulator()
    #    snr_folders[snr] = sim.folder_path

    return figure_folder, sim.folder_path

def run_snr_exp(snrs = None, pos_noise_powers = None, noise_type="gaussian", pos_noise_type="gaussian"):
    #base_fig_path = pathlib.Path(__file__).parent.joinpath("figs")
    #figure_folder = utils.get_unique_folder("figs_", base_fig_path, detailed_naming=False)
    #figure_folder.mkdir(parents=True, exist_ok=True)
    #print(f"Saving figures to {figure_folder}")
    rng = np.random.default_rng(123456356)

    exp_metadata = {}
    if snrs is None:
        snrs = [40, 30, 20, 10, 0]
    if not isinstance(snrs, (list, tuple, np.ndarray)):
        snrs = [snrs]
    if pos_noise_powers is None:
        pos_noise_powers = np.array([1e-2, 1e-1])**2
    if not isinstance(pos_noise_powers, (list, tuple, np.ndarray)):
        pos_noise_powers = np.array([pos_noise_powers])
    #num_mics = 12
    #for noise_p in [1e-4, 1e-3, 1e-2, 1e-1, 1]:

    base_folder, sim_folder = generate_signals_2d()

    exp_metadata["snr"] = snrs.tolist() if isinstance(snrs, np.ndarray) else snrs
    exp_metadata["pos_noise_powers"] = pos_noise_powers.tolist() if isinstance(pos_noise_powers, np.ndarray) else pos_noise_powers
    with open(base_folder.joinpath("metadata.json"), "w") as f:
        json.dump(exp_metadata, f, indent = 4)

    run_single_exp(sim_folder, snrs, pos_noise_powers, noise_type, pos_noise_type, rng)

    #with open(base_folder / "snr_folders.json", "w") as f:
    #    snrf = {key : str(val) for key, val in snr_folders.items()}
     #   json.dump(snrf, f, indent=4)

    #for snr_val, in snrs:
    #    rng = np.random.default_rng(123456356)
    #    run_single_exp(folder, snr_val, pos_noise_powers, noise_type, pos_noise_type, rng)

    snr_exp_post_process(base_folder)


def run_snr_exp_from_folder(base_folder):
    # UNCOMMENT TO CHANGE THE NOISE POWERS
    #pos_noise_powers = np.array([1e-2, 1e-1, 3e-1])**2
    #with open(base_folder / "metadata.json", "r") as f:
    #    exp_metadata = json.load(f)
    #exp_metadata["pos_noise_powers"] = pos_noise_powers.tolist()
    #with open(base_folder / "metadata.json", "w") as f:
    #    json.dump(exp_metadata, f, indent = 4)

    with open(base_folder / "metadata.json", "r") as f:
        exp_metadata = json.load(f)

    with open(base_folder / "snr_folders.json", "r") as f:
        snr_folders = json.load(f)

    for snr, folder in snr_folders.items():
        folder = pathlib.Path(folder)
        rng = np.random.default_rng(123456356)
        run_single_exp(folder, exp_metadata["pos_noise_powers"], rng)

    snr_exp_post_process(base_folder)


def _load_metric(main_path, metric_name):
    all_mse = {}
    for pth in main_path.iterdir():
        if pth.is_dir() and pth.stem.startswith("figs_"):
            if (pth / metric_name).exists():
                with open(pth.joinpath(metric_name), "r") as f:
                    mse_db = json.load(f)
                    for est_name, mse_val in mse_db.items():
                        if est_name not in all_mse:
                            all_mse[est_name] = []
                        all_mse[est_name].append(mse_val)
    return all_mse


def snr_exp_post_process(main_path):
    with open(main_path.joinpath("metadata.json"), "r") as f:
        exp_metadata = json.load(f)
    snr = np.array(exp_metadata["snr"])
    snr_strings = [f"inf" if np.isinf(snr_val) else f"{int(snr_val)}" for snr_val in snr]
    pos_noise_powers = np.array(exp_metadata["pos_noise_powers"])

    mse = _load_metric(main_path, "mse_db.json")

    potential_methods = ("moving omni", "spatial spectrum", "kernel interpolation", "nearest neighbour")
    used_methods = []
    for est_name in mse.keys():
        for method_name in potential_methods:
            if est_name.startswith(method_name) and method_name not in used_methods:
                used_methods.append(method_name)
    mse_snr = {}
    mse_pos_noise = {}
    for method_name in used_methods:
        mse_snr[method_name] = {f"{method_name} pos_noise:{pnp}" : np.array([mse[f"{method_name} snr:{snr_val} pos_noise:{pnp}"] for snr_val in snr_strings]) for pnp in pos_noise_powers}
        mse_pos_noise[method_name] = {f"{method_name} snr:{snr_val}" : np.array([mse[f"{method_name} snr:{snr_val} pos_noise:{pnp}"] for pnp in pos_noise_powers]) for snr_val in snr_strings}
        mse_snr[method_name] = {key : np.squeeze(val) for key, val in mse_snr[method_name].items() if val.ndim >= 1}
        mse_pos_noise[method_name] = {key : np.squeeze(val) for key, val in mse_pos_noise[method_name].items() if val.ndim >= 1}
    
    mse_ref_methods = {est_name : mse_vals for est_name, mse_vals in mse.items() if est_name.startswith("noise free")}
    #{est_name : mse_vals for est_name, mse_vals in mse.items() if est_name not in mse_moving_omni_snr and est_name not in mse_moving_omni_pos_noise}



    fig, ax = plt.subplots(1,1, figsize=(6,6))
    for method_name, sub_dict in mse_snr.items():
        for details, mse_vals in sub_dict.items():
            ax.plot(snr, mse_vals, label=f"{details}")

    for est_name, mse_vals in mse_ref_methods.items():
        #horiontal line
        ax.plot(snr, [mse_vals[0]]*len(snr), label=est_name)#, linestyle="--")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE (dB)")
    ax.legend()
    #ax.set_xscale("log")
    ax.set_title("MSE vs SNR")
    aspplot.set_basic_plot_look(ax)
    aspplot.save_plot(OUTPUT_METHOD, main_path, "mse_vs_snr")

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    for method_name, sub_dict in mse_pos_noise.items():
        for details, mse_vals in sub_dict.items():
            ax.plot(np.log10(pos_noise_powers + 1e-10), mse_vals, label=details)

    for est_name, mse_vals in mse_ref_methods.items():
        #horiontal line
        ax.plot(np.log10(pos_noise_powers + 1e-10), [mse_vals[0]]*len(pos_noise_powers), label=est_name)#, linestyle="--")
    ax.set_xlabel("pos_noise power (log10)")
    ax.set_ylabel("MSE (dB)")
    ax.legend()
    #ax.set_xscale("log")
    ax.set_title("MSE vs pos noise power")
    aspplot.set_basic_plot_look(ax)
    aspplot.save_plot(OUTPUT_METHOD, main_path, "mse_vs_pos_noise")

    #with open(main_path.joinpath("mse_all_db.json"), "w") as f:
    #    json.dump({est_name : mse_vals.tolist() for est_name, mse_vals in mse.items()}, f, indent = 4)

def plot_pos_for_paper(pos_moving, pos_mic, pos_loudspeaker, samplerate, figure_folder, output_method="pdf", name_suffix=""):
    fig, ax = plt.subplots(1,1, figsize=(10,10))

    
    jump_idx = _find_jump_idx(pos_moving)
    ax.plot(pos_moving[:jump_idx+1,0], pos_moving[:jump_idx+1,1], marker="x", label="moving mic 1", linestyle="None")
    ax.plot(pos_moving[jump_idx+1:,0], pos_moving[jump_idx+1:,1], marker="x", label="moving mic 2", linestyle="None")

    ax.plot([pos_moving[jump_idx,0], pos_moving[jump_idx+1,0]], [pos_moving[jump_idx,1], pos_moving[jump_idx+1,1]], label="jump")

    ax.plot(pos_moving[0,0], pos_moving[0,1], marker="o", label="start", linestyle="None")
    ax.plot(pos_moving[-1,0], pos_moving[-1,1], marker="s", label="end", linestyle="None")

    ax.plot(pos_moving[np.arange(1, pos_moving.shape[0], samplerate),0], pos_moving[np.arange(1, pos_moving.shape[0], samplerate),1], label="second markers", marker="x", linestyle="None")

    ax.plot(pos_mic[:,0], pos_mic[:,1], label="microphones", marker="o", linestyle="None")
    ax.plot(pos_loudspeaker[:,0], pos_loudspeaker[:,1], label="loudspeakers", marker="s", linestyle="None")
    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis('equal')
    aspplot.set_basic_plot_look(ax)
    aspplot.save_plot(output_method, figure_folder, f"positions{name_suffix}")

    latexutil.reduceSize(figure_folder / f"positions{name_suffix}" / f"positions{name_suffix}-000.dat", decimation=10, precision=3)
    latexutil.reduceSize(figure_folder / f"positions{name_suffix}" / f"positions{name_suffix}-001.dat", decimation=10, precision=3)

def _find_jump_idx(pos_moving):
    traj_diff = np.linalg.norm(np.diff(pos_moving, n = 1, axis=0), axis=1)
    median_traj_diff = np.median(traj_diff)
    jump_idx = np.argmax(traj_diff)

    if traj_diff[jump_idx] < 100 * median_traj_diff:
        print(f"Likely that jump index is incorrect, median diff: {median_traj_diff}, jump diff: {traj_diff[jump_idx]}")
    return jump_idx



if __name__ == "__main__":
    OUTPUT_METHOD = "tikz"

    #data_fdr = pathlib.Path("C:/research/papers/2025_forum_acusticum_moving_microphone_dataset/code/figs/figs_2025_03_31_20_06_0/figs_2025_03_31_20_06_0")
    #run_single_exp(data_fdr, [40, 30, 20, 10, 0], [0], "real", "angles", None)
    #run_snr_exp_from_folder(data_fdr.parent)
    #snr_exp_post_process(data_fdr.parent)

    #data_fdr = pathlib.Path("C:/research/papers/2025_forum_acusticum_moving_microphone_dataset/code/figs/figs_2025_03_31_17_46_0/figs_2025_03_31_17_46_0")
    #run_single_exp(data_fdr, [40, 30, 20, 10, 0], [0], "real", "angles", None)
    #run_snr_exp_from_folder(data_fdr.parent)
    #snr_exp_post_process(data_fdr.parent)

    #run_snr_exp([np.inf], np.array([0.0]), "real", "angles")
    # main_path = pathlib.Path("C:/research/papers/2025_forum_acusticum_moving_microphone_dataset/code/figs/figs_2025_03_31_17_46_0")
    # snr_exp_post_process(main_path)
    # main_path = pathlib.Path("C:/research/papers/2025_forum_acusticum_moving_microphone_dataset/code/figs/figs_2025_03_31_20_06_0")
    # snr_exp_post_process(main_path)

    run_snr_exp([-10, 0, 10, 20, 30, 40, 50, 60], [0.0], "real", "angles")
    run_snr_exp([np.inf], np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), "real", "angles")
    

    #run_snr_exp([50, 60], [0.0], "real", "angles")

    #snr_exp_post_process(pathlib.Path("C:/research/papers/2025_forum_acusticum_moving_microphone_dataset/code/figs/figs_2025_03_13_19_47_0"))

    #generate_snr_experiment_data()
    #multi_speed_experiment(sr)
    #fig_folder = exis.generate_signals_circular_updownsample(800, 2, 1)
    #fig_folder = generate_signals_2d()
    #fig_folder = exis.generate_signals_circular(sr, 1, 10)
    #fig_folder = pathlib.Path("C:/research/research_documents/202401_moving_mic_measurements/code_sim/figs/figs_2024_03_25_10_18_0")
    #main(fig_folder)
    #fig_folder = exis.generate_signals_circular(sr, 2, 20)
    #run_exp(fig_folder)
    #fig_folder = pathlib.Path("c:/research/research_documents/202305_moving_mic_spatial_cov_estimation/code/figs/figs_2023_08_02_12_44_0")
    #fig_folder = pathlib.Path("c:/research/research_documents/202305_moving_mic_spatial_cov_estimation/code/figs/figs_2023_07_04_15_31_0")
    

    #from pyinstrument import Profiler
    #profiler = Profiler()
    #profiler.start()
    #run_exp(fig_folder)

    #profiler.stop()
    #profiler.print()