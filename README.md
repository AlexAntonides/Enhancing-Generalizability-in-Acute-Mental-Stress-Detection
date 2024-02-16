<a name="readme-top"></a>
<!-- PROJECT SHIELDS -->
[![Python Version][python-shield]][python-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://alex-antonides.com/">
    <svg stroke="white" fill="white" stroke-width="0" height="5em" width="5em" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 135.96 117.81"><path d="M15.42,117.73,0,117.76,28.05,68.91s31,.15,31-.19S43.74,42.29,43.74,41.79,68.22,0,68.22,0l34.54,59.66-13.83-.08L68.26,24.24,57.8,40.87,72.88,68.74l35,.09,28.12,49H123.23l-7.76-12.18-14.71-25L35,80.74,21.25,106Z"></path></svg>
  </a>

  <h3 align="center">Master's Thesis</h3>

  <p align="center">
    Repository for the code I've written for my Master's Thesis.
    <br />
    <a href="https://github.com/AlexAntonides/stress-in-action/tree/master/sia/docs/structure.md"><strong>Explore the docs »</strong></a>
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

This is still a work-in-progress. This is the code I've written so far for my Master's Thesis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python v3.10+

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Preparation
1.  **Prepare your dataset**.\
The program expects a dataset with columns [signal, bool]. \
This dataset can be made using the `Pipeline` as seen in `pipeline.ipynb`, or with the following, 
```py
sia.Pipeline() \
    .data(read_csv('./data/preprocessed_data/*.csv')) \
    .reduce(reduce) \
    .postprocess(encode_category) \
    .to(write_csv('./data/model/[0-9]{5}.csv'))
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

_A step by step series of examples that tell you how to get the code running._

1. **Install The Packages**. \
Open the terminal in the root directory and execute the following command,
```bash
pip[3] install -r requirements.txt
```

2. **Train**. \
You can train by using the `train.py` script, by executing the following command,
```bash
python[3] train.py sia.models.[model_name] ./data/model/*.csv
```
Or alternatively, use the `--test` flag to train on one file for testing purposes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- FREQUENTLY ASKED QUESTIONS -->
## Frequently Asked Questions

1. **Where are the models defined?**
> The base of the models are currently stored in `sia.models.*`. These layers will eventually be used in `train.py` where the train model (inheriting the given model) will use the given model's layer structure by calling `forward(self, x)` (or `self(x)` in this case).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-shield]: https://img.shields.io/badge/3.10%2B-yellow?style=for-the-badge&logo=python&logoColor=white&label=python&labelColor=blue
[python-url]: https://www.python.org/