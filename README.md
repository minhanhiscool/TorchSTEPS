# TorchSTEPS

☔ A WIP project using PySTEPS and PyTorch to accurately nowcast the weather

## Overview

This code uses Convolutional LSTM (ConvLSTM), which is specified in this [paper](https://arxiv.org/pdf/1506.04214v2) (We are using the [PyTorch implementation](https://github.com/ndrplz/ConvLSTM_pytorch) of the ConvLSTM paper) to predict the weather using the past 20 minutes of raw radar data and estimates of three classical models (Extrapolation, STEPS, and ANVIL)

## Requirements

- Python 3.8+
- Packages listed in requirements.txt

## Installation

Clone the repository by running:
```
git clone https://github.com/minhanhiscoolish/TorchSTEPS
```
(Optionally) Create a virtual environment in the cloned repository:
```
cd TorchSTEPS
python3 -m venv venv
```
Install the required packages:
```
pip install -r requirements.txt
```

## Usage (WIP)
```
python3 src/main.py
```
## Contributing

We welcome contributions from everyone! Whether it's fixing bugs, improving code clarity, etc.

### How to contribute

Clone the repository, then create a new branch for your changes:
```
git clone https://github.com/minhanhiscoolish/TorchSTEPS
git checkout -b feature/my-branch
```
Make your changes. Make sure to follow the coding style.

Test your changes, then commit your changes:
```
git commit -a -m "Commit message"
```
Push your changes to another branch
```
git push origin feature/my-branch
```
To switch branches, use:
```
git checkout main
git checkout feature/my-branch
```
> [!WARNING]
> You should never push your changes to the main branch!

## License
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
