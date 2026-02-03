import json

import matplotlib.pyplot as plt
import numpy as np

import aspcore.fouriertransform as ft
import aspcore.utilities as utils

import aspcol.soundfieldestimation as sfe
import aspcol.plot as aspplot

import experiment_functions

def main_reg_parameter():
    info, pos, signals, rir, pos_moving, sig_moving, loudspeaker_moving, noise_moving, pos_image, rir_eval_freq, wave_num, freqs, figure_folder = experiment_functions.load_exp_data(output_method=OUTPUT_METHOD)

    mic_idxs = np.array([1, 5, 20, 27, 33, 39, 54, 57])
    pos_mic = pos["mic"][mic_idxs,:]
    rir_mic_freq = rir_eval_freq[:,mic_idxs]
    pos_eval = pos["mic"]
    
    noise_power = np.mean(noise_moving**2)
    print(f"noise power: {noise_power}")
    regularization_all = 10.0**np.arange(-8, 0, dtype=float) #np.logspace(-5, -1, 9)
    estimates = {}
    estimates_image = {}

    for regularization in regularization_all:
        estimates[f"kernel interpolation {regularization}"] = sfe.est_ki_diffuse_freq(rir_mic_freq, pos_mic, pos_eval, wave_num, regularization)
        estimates_image[f"kernel interpolation {regularization}"] = sfe.est_ki_diffuse_freq(rir_mic_freq, pos_mic, pos_image, wave_num, regularization)

        estimates[f"moving omni {regularization}"], regressor, _ = sfe.inf_dimensional_shd_dynamic(sig_moving, pos_moving, pos_eval, loudspeaker_moving, info["samplerate"], info["c"], regularization, verbose=True)
        estimates[f"spatial spectrum {regularization}"] = sfe.est_spatial_spectrum_dynamic(sig_moving, pos_moving, pos_eval, loudspeaker_moving, info["samplerate"], info["c"], regularization, verbose=False)

        estimates_image[f"moving omni {regularization}"] = sfe.estimate_from_regressor(regressor, pos_moving, pos_image, wave_num) 
        estimates_image[f"spatial spectrum {regularization}"] = sfe.est_spatial_spectrum_dynamic(sig_moving, pos_moving, pos_image, loudspeaker_moving, info["samplerate"], info["c"], regularization, verbose=False)

    aspplot.soundfield_estimation_comparison(pos_eval, estimates, rir_eval_freq, freqs, figure_folder, shape="rectangle", output_method=OUTPUT_METHOD, images=estimates_image, image_true=None, pos_image=pos_image, num_examples = 16)

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
    info, pos, signals, rir, pos_moving, sig_moving, loudspeaker_moving, noise_moving, pos_image, rir_eval_freq, wave_num, freqs, figure_folder = experiment_functions.load_exp_data(output_method=OUTPUT_METHOD)

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

    aspplot.soundfield_estimation_comparison(pos_eval, estimates, rir_eval_freq, freqs, figure_folder, shape="rectangle", output_method=OUTPUT_METHOD, images=estimates_image, image_true=None, pos_image=pos_image, num_examples = 16)

if __name__ == "__main__":
    OUTPUT_METHOD = "pdf"
    #fig_folder = pathlib.Path(__file__).parent / "figs" / "figs_2025_03_20_11_03_0"
    #_reg_parameter_plot(fig_folder)
    main()
    main_reg_parameter()