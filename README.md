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

  <h3 align="center">Enhancing Generalizability in Acute Mental Stress Detection: <br/> A Machine Learning Approach Leveraging ECG from a Novel Dataset</h3>

  <p align="center">
    This repository houses the codebase to predict stress solely on the electrical activity of the heart using machine learning techniques.
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

<!-- ABOUT THE PROJECT -->
## About The Project
This project aims to predict the stress of an unknown person soley on the electrical activity of the heart. This is achieved through training machine learning models on electrocardiogram (ECG) recordings.

### Stress-in-Action (SiA)-Kit
The SiA-Kit is a package that provides human-readable and extendable pipelines to repeat the experiments described in the project. The builder pattern that is used throughout the pipelines, makes it easier to understand what is happening behind the scenes. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python v3.9+

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Preparation
1. **Install The Packages**. \
Open the terminal in the root directory and execute the following command, 
```bash
pip[3] install -r requirements.txt
```

### Preprocessing
_A step by step series of examples that tell you how to get the data cleaned._

1.  **Prepare your dataset**.\
The directories that were used throughout the experiment can be found in `data/`, which is separated into four parts, `raw` for the raw data, `cleaned` for the cleaned data, `features` for the manually extracted features used to build the machine learning models, and `signal` which can be used by a neural network.\
The use of these directories are **optional**, any directories can be used, as long as the paths are changed accordingly.\
The dataset can be prepared by using the pipeline found in `1_preprocessing.ipynb`, or by using the methods that can be found in the SiA-kit, like so,

```py
sia.Preprocessing() \
    .data(read_csv('<in_path>/*.csv')) \
    .process(neurokit()) \
    .to(write_csv('<out_path>/[0-9]{5}.csv'))
```

2. **Extract the features**.\
The features can be extracted by using the pipeline found in `2_feature_extraction.ipynb`, or by using the methods that can be found in the SiA-kit, like so,

```py
sia.Segmenter() \
    .data(read_csv('<in_path>/*.csv')) \
    .segment(SlidingWindow(WINDOW_SIZE, int(STEP_SIZE))) \
        .extract(hrv([Statistic.MEAN, Statistic.CVSD, Statistic.NN20]))
    .to(write_csv('<out_path>/[0-9]{5}.csv'))
```

3. **Build the model**.\
The model can be built by using the pipeline found in `3_training.ipynb`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- FREQUENTLY ASKED QUESTIONS -->
## Frequently Asked Questions

1. **Where are the experiments?**\
For better readability, this repository has been overhauled. The notebooks with the experiments can be found in `/deprecated`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-shield]: https://img.shields.io/badge/3.9%2B-yellow?style=for-the-badge&logo=python&logoColor=white&label=python&labelColor=blue
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/2.1%2B-orange?style=for-the-badge&logo=pytorch&logoColor=orange&label=PyTorch&labelColor=white
[pytorch-url]: https://pytorch.org/