# Rotating Moving Microphone Sound field (RoMMS) dataset
## Description
The dataset is intended to be used for evaluation sound field estimation methods using moving microphones. The dataset consists of sound field measurements made with 60 stationary microphones on a square grid, and two moving microphones moving along concentric circles around the square grid. The sound field measurements consists of simultaneously recorded microphone signals, loudspeaker signals and positions of the microphones. 


<img src="https://github.com/user-attachments/assets/651d0454-7811-40e0-b13f-844cf588dded" height="180" alt="Positions of microphones and loudspeakers">
<img src="https://github.com/user-attachments/assets/951e283b-c37e-47a6-97a6-560b924385e4" height="180" alt="Stationary microphone array">
<img src="https://github.com/user-attachments/assets/683acb43-60f0-4c0f-9d53-0d27c740be0b" height="180" alt="Moving microphones">



## Download
The dataset can be downloaded from Zenodo at [doi.org/10.5281/zenodo.15124905](https://doi.org/10.5281/zenodo.15124905)

## License
The dataset is available under [Creative Commons 4.0 Attribution License](LICENSE)

If you use the dataset in your research, please cite the following paper.\
**Experimental validation of sound field estimation methods using moving microphones**\
*J. Brunnström, M. B. Møller, T. van Waterschoot, M. Moonen, and J. Østergaard*
```
@inproceedings{brunnstromExperimental2025,
  title={Experimental validation of sound field estimation methods using moving microphones}, 
  author={Brunnström, Jesper and Møller, Martin Bo and van Waterschoot, Toon and Moonen, Marc and Østergaard, Jan},
  year = {submitted},
  langid = {english},
  booktitle = {Proceedings of Forum Acusticum},
}
```

## Usage
The functions in load_dataset comes with a number of parameters, which represents options available in the dataset. The following is a description of the parameters for load_dataset.load(). The function load_dataset.load_noise() has analogous parameters. More technical documentation regarding data formats and such can be found in the docstring. 

### room
`Valid options are "a", "b", "c", "d"`

The dataset was recorded in the same room, but under 4 different acoustic conditions, created by varying the number of absorbent acoustic panels on the walls. These acoustic conditions are referred to as room A, B, C and D,
listed from most to fewest acoustic panels, hence from shortest to longest reverberation time. 


### seq_len_ms
`Valid options are 500, 1000`

The period of the periodic sweep should be chosen as equal length or longer than the room impulse response (RIR), otherwise a wraparound error is incurred. Recordings were made using a period length of 500 ms as well as 1000 ms. Considering the reverberation time, a period length of 1000 ms appears clearly sufficient, while
500 ms could incur a small error. However, because the RIRs will have decayed close to 60 dB after 500 ms, unless a very high signal-to-noise ratio is achieved, the additional error is likely negligible.

### max_freq
`Valid options are 500, 1000, 2000, 4000`

The measurements are made with a periodic perfect sweep signal, the frequency of which increases linearly from 0 to $f_{\text{max}}$. The dataset contains recordings using a maximum frequency of 500, 1000, 2000, and 4000 Hz. 

### speed
`Valid options are "fast", "slow"`

There are two different speeds at which the moving microphone was recorded. The speed is not exactly constant for either option, but similar for all realizations with the same option selected. The mean speed over all recordings for the fast setting are 0.334 m/s and 0.376 m/s, and for the slow setting 0.186 m/s and 0.206 m/s, for the inner and outer microphone respectively. 

### downsampled
`Valid options are True, False`

The periodic sweep signal has no energy above max_freq, hence there is rarely a need for a sampling rate above $`2 \cdot`$max_freq. If True, the data is downsampled to a sampling rate of $`2 \cdot`$max_freq. If False, the original sampling rate of 48 kHz is used. 

### alt_array
`Valid options are True, False`

The stationary array is recorded for room B, C and D in an addition location, which is the original position shifted -1 meter along the x-axis. If True, the measurements from the alternative position is returned for the stationary microphones. 

### dataset_folder
`Valid options are a string or pathlib.Path`

The parameter should point to the location on your computer where the RoMMS dataset downloaded from Zenodo is placed. In the example given below, the dataset_folder parameter should be given as "path/to/parent/folder/romms_dataset". 
```bash
└── path/to/parent/folder
    └── romms_dataset
        ├── room_a
        │   └── ...
        ├── room_b
        │   └── ...
        ├── room_c
        │   └── ...
        ├── room_d
        │   └── ...
        ├── positions.npz
        ├── positions_alt.npz
        └── rec_info.json
```

## Details
![rir_time_animation](https://github.com/user-attachments/assets/f17b02c8-61a3-4590-9921-6b8193d13b69)


### Room near end 
<img src="https://github.com/user-attachments/assets/aafbcba0-9f9c-4cef-b80d-f2595decac35" height="200">
<img src="https://github.com/user-attachments/assets/2bb4b850-ef92-403b-9c9c-732344f2117a" height="200">

### Room far end
<img src="https://github.com/user-attachments/assets/ef842646-f9a4-4450-9c55-5b4b0522c41d" height="200">
<img src="https://github.com/user-attachments/assets/d7d4961a-3f7f-4832-8671-66aa33cf0592" height="200">
<img src="https://github.com/user-attachments/assets/3edfd384-71a2-4af6-b950-f3b078d8ec8a" height="200">






