import pathlib
import json

import matplotlib.pyplot as plt
import numpy as np

import aspcore.fouriertransform as ft
import aspcore.utilities as utils

import aspcol.soundfieldestimation as sfe
import aspcol.plot as aspplot

import exp_funcs_extra

def main_reg_parameter():
    info, pos, signals, rir, pos_moving, sig_moving, loudspeaker_moving, noise_moving, pos_image, rir_eval_freq, wave_num, freqs, figure_folder = exp_funcs_extra.load_exp_data(output_method=OUTPUT_METHOD)

    mic_idxs = np.array([1, 5, 20, 27, 33, 39, 54, 57])
    pos_mic = pos["mic"][mic_idxs,:]
    rir_mic_freq = rir_eval_freq[:,mic_idxs]
    pos_eval = pos["mic"]
    
    # FREQ DOMAIN REGULARIZATION FROM NOISE MOVING
    # noise_freq = np.stack([ft.fft(noise_moving[i*info["seq_len"]:(i+1)*info["seq_len"]]) for i in range(noise_moving.shape[-1] // info["seq_len"])], axis=-1)
    # noise_freq_power = np.mean(np.abs(noise_freq)**2, axis=-1)
    # noise_freq_mean_power = np.mean(noise_freq_power) # equals seq_len * noise_power
    # FREQ DOMAIN REGULARIZATION FROM NOISE STATIONARY
    #noise_freq = np.stack([ft.fft(signals["noise_stationary"][:,i*info["seq_len"]:(i+1)*info["seq_len"]]) for i in range(signals["noise_stationary"].shape[-1] // info["seq_len"])], axis=-1)
    #noise_freq_power = np.mean(np.abs(noise_freq)**2, axis=(1,2))
    #noise_freq_mean_power = np.mean(noise_freq_power) # equals seq_len * noise_power
    # SIMPLEST CALCULATION OF NOISE POWER

    #stationary_noise_power = np.mean(signals["noise_stationary"]**2)
    #stationary_noise_power_freq = stationary_noise_power * info["seq_len"]
    #with open(figure_folder / "regulation_freq.json", "w") as f:
    #    json.dump({"regulation_freq" : float(stationary_noise_power_freq), "stationary_noise_power" : float(stationary_noise_power)}, f)


    noise_power = np.mean(noise_moving**2)
    print(f"noise power: {noise_power}")
    regularization_all = 10.0**np.arange(-8, 0, dtype=float) #np.logspace(-5, -1, 9)
    estimates = {}
    estimates_image = {}

    for regularization in regularization_all:
        estimates[f"kernel interpolation {regularization}"] = sfe.est_ki_diffuse_freq(rir_mic_freq, pos_mic, pos_eval, wave_num, regularization)
        estimates_image[f"kernel interpolation {regularization}"] = sfe.est_ki_diffuse_freq(rir_mic_freq, pos_mic, pos_image, wave_num, regularization)

        # print (f"omni estimation")
        estimates[f"moving omni {regularization}"], regressor, _ = sfe.inf_dimensional_shd_dynamic(sig_moving, pos_moving, pos_eval, loudspeaker_moving, info["samplerate"], info["c"], regularization, verbose=True)
        estimates[f"spatial spectrum {regularization}"] = sfe.est_spatial_spectrum_dynamic(sig_moving, pos_moving, pos_eval, loudspeaker_moving, info["samplerate"], info["c"], regularization, verbose=False)

        
        # print (f"omni estimation")
        estimates_image[f"moving omni {regularization}"] = sfe.estimate_from_regressor(regressor, pos_moving, pos_image, wave_num) 
        estimates_image[f"spatial spectrum {regularization}"] = sfe.est_spatial_spectrum_dynamic(sig_moving, pos_moving, pos_image, loudspeaker_moving, info["samplerate"], info["c"], regularization, verbose=False)
    
    abs_folder =  figure_folder / "abs_only"
    abs_folder.mkdir(parents=True, exist_ok=True)
    aspplot.soundfield_estimation_comparison(pos_eval, {est_name : np.abs(est) for est_name, est in estimates.items()}, np.copy(np.abs(rir_eval_freq)), freqs, abs_folder, output_method=OUTPUT_METHOD)

    aspplot.soundfield_estimation_comparison(pos_eval, estimates, np.copy(rir_eval_freq), freqs, figure_folder, shape="rectangle", output_method=OUTPUT_METHOD, images=estimates_image, image_true=None, pos_image=pos_image, num_examples = 16)

    image_scatter_freq_response(estimates_image, freqs, pos_image, figure_folder, plot_name="eval", marker_size=40)
    image_scatter_freq_response({"measured" : ft.rfft(rir)}, freqs, pos_eval, figure_folder, plot_name="_measured_rirs", marker_size=1250)

    _reg_parameter_plot(figure_folder)

def _reg_parameter_plot(fig_folder):
    with open(fig_folder / "mse_db.json", "r") as f:
        mse = json.load(f)

    potential_algo_names = ("kernel interpolation", "moving omni", "spatial spectrum")
    algo_names = []
    reg_values = {}
    for name, mse_val in mse.items():
        for algo_name in potential_algo_names:
            if name.startswith(algo_name):
                if algo_name not in reg_values:
                    reg_values[algo_name] = []
                    reg_values[f"{algo_name}_reg"] = []
                    algo_names.append(algo_name)
                reg_value = float(name.split(" ")[-1])
                reg_values[f"{algo_name}_reg"].append(reg_value)
                reg_values[algo_name].append(mse_val)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for name in algo_names:
        ax.plot(reg_values[f"{name}_reg"], reg_values[name], label=name)
    ax.set_xscale("log")
    ax.set_ylabel("MSE (dB)")
    ax.set_xlabel("Regularization value")
    ax.legend()
    utils.set_basic_plot_look(ax)
    utils.save_plot(OUTPUT_METHOD, fig_folder, "reg_parameter_plot")


def main():
    info, pos, signals, rir, pos_moving, sig_moving, loudspeaker_moving, noise_moving, pos_image, rir_eval_freq, wave_num, freqs, figure_folder = exp_funcs_extra.load_exp_data(output_method=OUTPUT_METHOD)

    mic_idxs = np.array([1, 5, 20, 27, 33, 39, 54, 57])
    pos_mic = pos["mic"][mic_idxs,:]
    rir_mic_freq = rir_eval_freq[:,mic_idxs]
    pos_eval = pos["mic"]
    
    noise_power = np.mean(noise_moving**2)
    lambda_inv = 0.1 
    regularization_mo = noise_power * lambda_inv
    regularization_ss = noise_power * lambda_inv
    estimates = {}
    estimates["kernel interpolation"] = sfe.est_ki_diffuse_freq(rir_mic_freq, pos_mic, pos_eval, wave_num, regularization_mo)
    print (f"omni estimation")
    estimates["moving omni"], regressor, _ = sfe.inf_dimensional_shd_dynamic(sig_moving, pos_moving, pos_eval, loudspeaker_moving, info["samplerate"], info["c"], regularization_mo, verbose=True)
    estimates["spatial spectrum"] = sfe.est_spatial_spectrum_dynamic(sig_moving, pos_moving, pos_eval, loudspeaker_moving, info["samplerate"], info["c"], regularization_ss, verbose=False)

    estimates_image = {}
    estimates_image["kernel interpolation"] = sfe.est_ki_diffuse_freq(rir_mic_freq, pos_mic, pos_image, wave_num, regularization_mo)
    print (f"omni estimation")
    estimates_image["moving omni"] = sfe.estimate_from_regressor(regressor, pos_moving, pos_image, wave_num) 
    estimates_image["spatial spectrum"] = sfe.est_spatial_spectrum_dynamic(sig_moving, pos_moving, pos_image, loudspeaker_moving, info["samplerate"], info["c"], regularization_ss, verbose=False)
    
    abs_folder =  figure_folder / "abs_only"
    abs_folder.mkdir(parents=True, exist_ok=True)
    aspplot.soundfield_estimation_comparison(pos_eval, {est_name : np.abs(est) for est_name, est in estimates.items()}, np.copy(np.abs(rir_eval_freq)), freqs, abs_folder, output_method=OUTPUT_METHOD)

    aspplot.soundfield_estimation_comparison(pos_eval, estimates, np.copy(rir_eval_freq), freqs, figure_folder, shape="rectangle", output_method=OUTPUT_METHOD, images=estimates_image, image_true=None, pos_image=pos_image, num_examples = 16)

    image_scatter_freq_response(estimates_image, freqs, pos_image, figure_folder, plot_name="eval", marker_size=40)
    image_scatter_freq_response({"measured" : ft.rfft(rir)}, freqs, pos_eval, figure_folder, plot_name="_measured_rirs", marker_size=1250)

def image_scatter_freq_response(ir_all_freq, freqs, pos, fig_folder, plot_name="", marker_size=200):
    """
    ir_all_freq is a dict, where each value is a ndarray of shape (num_freq, num_pos)
    freqs is a 1-d np.ndarray with all frequencies
    pos is a ndarray of shape (num_pos, 3)
    """
    num_freqs = freqs.shape[-1]
    num_example_freqs = 4
    idx_interval = num_freqs // (num_example_freqs+1)
    freq_idxs = np.arange(num_freqs)[idx_interval::idx_interval]

    for fi in freq_idxs:
        fig, axes = plt.subplots(len(ir_all_freq), 3, figsize=(15, len(ir_all_freq)*7), squeeze=False)
        for ax_row, (est_name, ir_val) in zip(axes, ir_all_freq.items()):
            #mse_val += 1e-6
            #mse_val = 10 * np.log10(mse_val)

            clr = ax_row[0].scatter(pos[:,0], pos[:,1], c=np.real(ir_val[fi,:]), s=marker_size, marker="s")
            cbar = fig.colorbar(clr, ax=ax_row[0])
            cbar.set_label('Real pressure')

            clr = ax_row[1].scatter(pos[:,0], pos[:,1], c=np.imag(ir_val[fi,:]), s=marker_size, marker="s")
            cbar = fig.colorbar(clr, ax=ax_row[1])
            cbar.set_label('Imag pressure')

            clr = ax_row[2].scatter(pos[:,0], pos[:,1], c=np.abs(ir_val[fi,:]), s=marker_size, marker="s")
            cbar = fig.colorbar(clr, ax=ax_row[2])
            cbar.set_label('Abs pressure')

            ax_row[0].set_title(f"Real: {est_name}")
            ax_row[1].set_title(f"Imag: {est_name}")
            ax_row[2].set_title(f"Abs: {est_name}")

            for ax in ax_row:
                #if "moving_mic" in pos:
                #    moving_mic.plot_moving_mic(pos["moving_mic"], ax)
                #    ax.legend(loc="lower left")
                #ax.legend(loc="lower left")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                
                ax.set_aspect("equal")
   
        utils.save_plot(OUTPUT_METHOD, fig_folder, f"image_scatter_freq_{freqs[fi]}Hz{plot_name}")



if __name__ == "__main__":
    OUTPUT_METHOD = "pdf"
    #fig_folder = pathlib.Path(__file__).parent / "figs" / "figs_2025_03_20_11_03_0"
    #_reg_parameter_plot(fig_folder)
    main()
    main_reg_parameter()