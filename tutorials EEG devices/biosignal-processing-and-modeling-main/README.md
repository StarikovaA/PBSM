# Biosignal Processing and Modeling Course Tutorials

Tutorials for the Biosignal Processing and Modeling course of SS23.

0. Setup
1. LSL Streams
2. Eye Blink Detection
3. Paradigm Design Using Pygame
4. EEG Data Analysis

Course instructors: Alireza Malekmohammadi, Nicolas Berberich

Teaching assistant: Karahan Yilmazer

## Installation
Download or clone the repository to your computer. You will see that each tutorial has its own dedicated folder. In some of the folders you will find fully working scripts, in others you will have to fill parts of the code fitting to your needs. All scripts are written in Python.

## Usage
It is recommended to use the Python scripts with [Visual Studio Code](https://code.visualstudio.com/). This way they can be run both as whole Python scripts and cell by cell with a Jupyter interactive window. The cells are separated from each other with the special comment `# %%`.

You can read more about setting up and using VS Code in the [Setting up the Software and Hardware](https://gitlab.lrz.de/neuro1/biosignal-processing-and-modeling/-/blob/main/T0-setup/Tutorial-Setting_Up.pdf) tutorial. **You are highly encouraged to go over this tutorial as it lays the groundwork for all the other tutorials.**

## Support
If you need assistance, Thursdays would be the best day to come to the lab. Please let me know in advance if you are planning to come.

-  Karahan Yilmazer: yilmazerkarahan@gmail.com

## Repository structure
```
biosignal-processing-and-modeling
├─ misc
│  ├─ Devices
│  │  ├─ Smarting
│  │  │  └─ Smarting 24 User Manual.pdf
│  │  └─ Unicorn
│  │     └─ Unicorn Hybrid Black User Manual.pdf
│  └─ requirements.txt
├─ README.md
├─ T0-setup
│  └─ Tutorial-Setting_Up.pdf
├─ T1-lsl-streams
│  ├─ lsl_inlet.py
│  ├─ lsl_outlet.py
│  ├─ lsl_print_sample.py
│  ├─ lsl_save_to_csv.py
│  ├─ plot_lsl_stream.py
│  ├─ read_xdf.py
│  ├─ real_time_lsl_filtering.py
│  └─ Tutorial-LSL_Streams.pdf
├─ T2-eye-blink-detection
│  ├─ eeg_eye_blink_detection.py
│  └─ Tutorial-Eye_Blink_Detection.pdf
└─ T3-designing-paradigms
   └─ maze_game_pygame.py

```