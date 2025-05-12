# TorchSTEPS

☔ A WIP project using PySTEPS and PyTorch to accurately nowcast the weather

## Overview

This code uses Convolutional LSTM (ConvLSTM), which is specified in this [paper](https://arxiv.org/pdf/1506.04214v2) (We are using the modified [PyTorch implementation](https://github.com/ndrplz/ConvLSTM_pytorch) of the ConvLSTM paper) to predict the weather using the past 20 minutes of raw radar data and estimates of three classical models (Extrapolation, STEPS, and ANVIL)

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
source venv/bin/activate
```
Install the required packages:
```
pip install -r requirements.txt
```

> [!NOTE]
> The requirements.txt **DOES NOT** have torch installed. It should be installed separately depending on your hardware
> ```
> pip install torch torchvision torchaudio
> ```

## Usage
First, collect data bulk from NEA website:
```
python3 src/pySTEPS/grabRadarSGBulk.py
```
Afterwards, you are ready to train:
```
cd src
python3 train.py
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

## Data Usage
This code can fetch and process data from the NEA website, which by its Terms of Use  
(see https://www.nea.gov.sg/corporate-functions/terms-of-use) **may only** be used for **personal, internal, non‑commercial**  
or **informational** purposes (Clause 4.3).

**We do NOT distribute NEA’s data**. You must download it yourself:
By running that, you confirm you’re using the data under NEA’s non‑commercial license.

## Contributors

See CONTRIBUTORS.md

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This project is licensed under the [MIT License](./LICENSE).



