<a name="readme-top"></a>
<!-- PROJECT SHIELDS -->
[![Python Version][python-shield]][python-url]
[![PyTorch Version][pytorch-shield]][pytorch-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://alex-antonides.com/">
    <svg stroke="white" fill="white" stroke-width="0" height="5em" width="5em" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 135.96 117.81"><path d="M15.42,117.73,0,117.76,28.05,68.91s31,.15,31-.19S43.74,42.29,43.74,41.79,68.22,0,68.22,0l34.54,59.66-13.83-.08L68.26,24.24,57.8,40.87,72.88,68.74l35,.09,28.12,49H123.23l-7.76-12.18-14.71-25L35,80.74,21.25,106Z"></path></svg>
  </a>

  <h3 align="center">Stress-in-Action</h3>

  <p align="center">
    This repository houses the codebase to predict stress solely on the electrical activity of the heart.
    <br />
    <a href="https://github.com/AlexAntonides/stress-in-action/blob/main/README.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/AlexAntonides/stress-in-action/issues">Report Bug</a>
    ·
    <a href="https://github.com/AlexAntonides/stress-in-action/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#frequently-asked-questions">Frequently Asked Questions</a>
    </li>
  </ol>
</details>

<!-- DEPRECATION -->
## Deprecation Notice
This repository is currently being refactored, to make the code more readable and less of a mess in general. This markdown file keeps track of the important code that is used in the experiments.

### 1. Feature Extraction
The features were extracted using the notebook in `deprecated/2_experiments_apr-jun/t.ipynb`.

### 2. Training of Machine Learning Models.
The models were trained in the notebook found in `deprecated/4_experiments_jul-aug/optuna_3.0.ipynb`.

### 3. Training of Neural Network.
The Command Line Interface (CLI) used to initiate the process can be found in `deprecated/4_experiments_jul-aug/train.py`.\
The CLI requires a dataset and model to run, which can be found in `deprecated/4_experiments_jul-aug/datasets/stepping_dataset.py` and `deprecated/4_experiments_jul-aug/models/time_series.py` respectively.\
The **[updated]** command used is, `py ./deprecated/4_experiments_jul-aug/train.py ./deprecated/4_experiments_jul-aug/models/time_series.py ./data/signal/mental_stress/*.csv --dataset=deprecated.4_experiments_jul-aug.datasets.stepping_dataset --batch_size=64 --window=10000 --epochs=200 --standardize`,\
or the more readable version, `py train.py <model> <data_files> --dataset=<dataset> --batch_size=<batch_size> --window=<window> --epochs=<epochs> --standardize`.  

### 4. Experiments
The experiments are scattered throughout the folders found in `deprecated/`, these include experiments to calculate certain features and to generate images for presentations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-shield]: https://img.shields.io/badge/3.9%2B-yellow?style=for-the-badge&logo=python&logoColor=white&label=python&labelColor=blue
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/2.1%2B-orange?style=for-the-badge&logo=pytorch&logoColor=orange&label=PyTorch&labelColor=white
[pytorch-url]: https://pytorch.org/