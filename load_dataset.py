import numpy as np
import pathlib
import json

import matplotlib.pyplot as plt
import matplotlib as mpl

DEFAULT_DATASET_FOLDER = pathlib.Path(__file__).parent.parent / "dataset"

def load(room="a",
        seq_len_ms=500,
        max_freq=1000,
        speed = "fast", 
        downsampled = True,
        alt_array = False,
        dataset_folder=DEFAULT_DATASET_FOLDER):
    """Function to load the dataset
    
    Parameters
    ----------
    room : str, one of {"a", "b", "c", "d"}
        refers to the room name. The dataset contains recordings from the same room
        with different configurations of the acoustic panels in the room. 
    max_freq : int, one of {500, 1000, 2000, 4000}
        refers to the maximum frequency of the loudspeaker signal. The impulse responses can
        not be calculated for frequencies higher than this value.
    seq_len : int, one of {500, 1000}
        refers to the sequence length in milliseconds
    speed : str, one of {'fast', 'slow', None}
        Selects to the speed of the moving microphone. If set to None, the data from the moving microphone will not be 
        loaded. A faster and slower recording is available for all settings
    downsampled : bool, default is True
        if true, the data will be downsampled to 2 * max_freq. There is no energy
        in the training signal above the max_freq, so downsampling to 2 * max_freq should be generally 
        useful, as it decreases the size of the data at minimal cost. 
    alt_array : bool, default is False
        If set to True, the function will load the alternative array as the stationary array. The alternative array is 
        the same shape of 60 microphones but moved 1 meter in the negative x direction. The alternative array is available 
        for rooms b, c, and d, but not for room a. 
    dataset_folder : str or pathlib.Path, optional
        If set, the function will load the dataset from this folder instead of the default one.
        The default folder is specified with regards to the load_dataset script

    Returns
    -------
    info : dict
        A dictionary containing the recording information.
        Contains the entries {'samplerate', 'num_periods_array', 'height', 'temp', 'c', 'seq_len'}
    pos : dict
        A dictionary containing the positions of the microphones and loudspeakers.
        Contains the entries {'mic_array', 'loudspeaker', 'mic_moving'}
    signals : dict
        A dict of np.ndarrays containing the microphone and loudspeaker signals
        Contains the entries {'mic_array0', 'mic_array1', 'loudspeaker_array0', 'loudspeaker_array1', 
        'mic_moving', 'loudspeaker_moving'}
    rir_data : ndarray of shape (60, rir_length)
        An array containing the room impulse responses associated with the microphone array.
        The rir_length is the same as the sequence length.
    """
    assert room in ["a", "b", "c", "d"], "Room must be one of {'a', 'b', 'c', 'd'}"
    assert max_freq in [500, 1000, 2000, 4000], "Max freq must be one of {500, 1000, 2000, 4000}"
    assert seq_len_ms in [500, 1000], "Sequence length must be one of {500, 1000}"
    assert speed in ["slow", "fast", None], "Speed must be one of {'slow', 'fast', None}"

    if room == "a" and alt_array:
        raise ValueError("The alternative array is not available for room a. Set alt_array to False to load the data from room a")
    
    if alt_array:
        alt_suffix = "_alt"
    else:
        alt_suffix = ""

    dataset_folder = pathlib.Path(dataset_folder)
    room_name = f"room_{room}"
    sess_name = f"{max_freq}Hz_{seq_len_ms}ms"
    room_folder = dataset_folder / room_name
    sess_folder = room_folder / sess_name
    noise_folder = room_folder / "noise"

    pos = np.load(dataset_folder / f"positions{alt_suffix}.npz")
    with open(dataset_folder / "rec_info.json", "r") as f:
        info = json.load(f)

    # Load data
    if downsampled:
        info["samplerate"] = int(max_freq*2)
        signals = np.load(sess_folder / f"array_signal{alt_suffix}_downsampled.npz")
        rir = np.load(sess_folder / f"array_rir{alt_suffix}_downsampled.npy")
        
        if speed is not None:
            moving_mic_data = np.load(sess_folder / f"moving_microphone_signals_{speed}_downsampled.npz")
    else:
        signals = np.load(sess_folder / f"array_signal{alt_suffix}.npz")
        rir = np.load(sess_folder / f"array_rir{alt_suffix}.npy")
        if speed is not None:
            moving_mic_data = np.load(sess_folder / f"moving_microphone_signals_{speed}.npz")
    info["seq_len"] = int(info["samplerate"] * seq_len_ms / 1000)

    # Convert the numpy file object to dictionaries
    pos = {key: val for key, val in pos.items()}
    signals = {key: val for key, val in signals.items()}

    # Merge moving microphone data into the dictionaries
    if speed is not None:
        pos["mic_moving"] = moving_mic_data["pos"]
        signals["mic_moving"] = moving_mic_data["mic"]
        signals["loudspeaker_moving"] = moving_mic_data["loudspeaker"]

    return info, pos, signals, rir

def load_noise(room="a",
                max_freq=1000,
                speed = "fast",
                downsampled = True,
                alt_array = False,
                dataset_folder=DEFAULT_DATASET_FOLDER):
    """Function to load the dataset
    
    Parameters
    ----------
    room : str, one of {"a", "b", "c", "d"}
        refers to the room name. The dataset contains recordings from the same room
        with different configurations of the acoustic panels in the room. 
    max_freq : int, one of {500, 1000, 2000, 4000}
        If downsampled == True, then the noise will be downsampled to 2 * max_freq. Otherwise it has
        no effect. 
    speed : str, one of {'fast', 'slow', None}
        Selects to the speed of the moving microphone. If set to None, the data from the moving microphone will not be 
        loaded. A faster and slower recording is available for all settings
    downsampled : bool, default is True
        if true, the data will be downsampled to 2 * max_freq. There is no intended energy
        in the training signal above the max_freq, so downsampling to 2 * max_freq should generally 
        useful, as it decreases the size of the data at essentially no cost. 
    dataset_folder : str or pathlib.Path, optional
        If set, the function will load the dataset from this folder instead of the default one.
        The default folder is specified with regards to the load_dataset script

    Returns
    -------
    info : dict
        A dictionary containing the recording information.
        Contains the entries {'samplerate', 'num_periods_array', 'height', 'temp', 'c', 'seq_len'}
    pos : dict
        A dictionary containing the positions of the microphones and loudspeakers.
        Contains the entries {'mic_array', 'mic_moving'}
    signals : dict
        A dict of np.ndarrays containing the microphone and loudspeaker signals
        Contains the entries {'mic_array', 'mic_moving', 'mic_moving_ambient'}
        The noise mic_moving_ambient is the noise from the moving microphones when they are not moving. It is recorded 
        at the starting position, meaning pos["mic_moving"][:,0,:]. 
    """
    assert room in ["a", "b", "c", "d"], "Room must be one of {'a', 'b', 'c', 'd'}"
    assert max_freq in [500, 1000, 2000, 4000], "Max freq must be one of {500, 1000, 2000, 4000}"
    assert speed in ["slow", "fast", None], "Speed must be one of {'slow', 'fast', None}"

    if room == "c" and speed == "slow":
        raise ValueError("The slow noise data for room c is not available. Set speed to 'fast' or None to load noise data for room c")
    
    if room == "a" and alt_array:
        raise ValueError("The alternative array is not available for room a. Set alt_array to False to load the data from room a")
    
    if alt_array:
        alt_suffix = "_alt"
    else:
        alt_suffix = ""

    dataset_folder = pathlib.Path(dataset_folder)
    room_name = f"room_{room}"
    room_folder = dataset_folder / room_name
    data_folder = room_folder / "noise"

    pos = np.load(dataset_folder / f"positions{alt_suffix}.npz")
    with open(dataset_folder / "rec_info.json", "r") as f:
        info = json.load(f)

    # Load data
    if downsampled:
        info["samplerate"] = int(max_freq*2)

        noise_array = np.load(data_folder / f"array_noise{alt_suffix}_{max_freq}.npy")
        if speed is not None:
            moving_mic_noise = np.load(data_folder / f"moving_microphone_noise_{speed}_{max_freq}.npz")
            moving_mic_ambient_noise = np.load(data_folder / f"moving_microphone_ambient_noise_{max_freq}.npy")
    else:
        noise_array = np.load(data_folder / f"array_noise{alt_suffix}.npy")
        if speed is not None:
            moving_mic_noise = np.load(data_folder / f"moving_microphone_noise_{speed}.npz")
            moving_mic_ambient_noise = np.load(data_folder / f"moving_microphone_ambient_noise.npy")

    # Convert the numpy file object to dictionaries
    pos_original = {key: val for key, val in pos.items()}
    pos = {}
    pos["mic_array"] = pos_original["mic"][:30,:]
    signals = {}
    signals["mic_array"] = noise_array

    # Merge moving microphone data into the dictionaries
    if speed is not None:
        signals["mic_moving_ambient"] = moving_mic_ambient_noise
        signals["mic_moving"] = moving_mic_noise["mic"]
        pos["mic_moving"] = moving_mic_noise["pos"]
    return info, pos, signals


def plot_pos(info, pos):
    fig, ax = plt.subplots(1,1, figsize = (12,9))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ax.scatter(pos["mic"][:,0], pos["mic"][:,1], marker="o", label = "Microphone array", edgecolors="black")
    for i in range(pos["mic"].shape[0]):
        ax.text(pos["mic"][i,0]+0.01, pos["mic"][i,1], f"{i}", fontsize=7, color="midnightblue", alpha=0.7)

    ax.scatter(pos["loudspeaker"][:,0], pos["loudspeaker"][:,1], marker="s", label = "Loudspeaker", edgecolors="black")

    
    if "mic_moving" in pos:
        num_periods = pos["mic_moving"].shape[1] // info["seq_len"]
        clr_map_values = np.linspace(0, 1, num_periods)
        cmap = mpl.colormaps['Reds']

        ax.plot(pos["mic_moving"][0,:,0], pos["mic_moving"][0,:,1], "-", label = f"Moving microphone", color = cmap(0.7))
        ax.scatter(pos["mic_moving"][0,::info["seq_len"],0], pos["mic_moving"][0,::info["seq_len"],1], c =clr_map_values,  marker="h", edgecolors="none", cmap='inferno', zorder=10)

        ax.plot(pos["mic_moving"][1,:,0], pos["mic_moving"][1,:,1], "-", color = cmap(0.7))

        for j in range(num_periods):
            ax.text(pos["mic_moving"][0,j*info["seq_len"],0] * 1.07, pos["mic_moving"][0,j*info["seq_len"],1] * 1.07, f"{j} s", fontsize=7, color=cmap(0.9), ha="center")

    ax.set_title("Microphone and loudspeaker positions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid()
    ax.legend()
    ax.axis("equal")

    plt.show()

if __name__ == "__main__":
    info, pos, array_data, rir_data = load(room="a",seq_len_ms=500, max_freq=1000, downsampled = True, dataset_folder=DEFAULT_DATASET_FOLDER)

    plot_pos(info, pos)
