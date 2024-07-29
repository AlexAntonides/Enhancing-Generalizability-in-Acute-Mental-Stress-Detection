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

## LSTM Structure
A basic example of the train configuration can be found [here](https://github.com/AlexAntonides/stress-in-action/blob/main/1_training.ipynb), which does as follows,

1. Considering the input is a collection of files (e.g. `participant_1.csv`, `participant_2.csv`, and so on.), a train/test/validation split is made on the files, such that,
```py
train_participants = ['participant_1', 'participant_3', ...]
val_participants = ['participant_2', 'participant_4', ...]
test_participants = ['participant_5', 'participant_6', ...]
```

2. The data is loaded in a [HuggingFace Dataset](https://huggingface.co/docs/datasets/en/index), through the [MultiParticipantDataModule](https://github.com/AlexAntonides/stress-in-action/blob/main/src/datamodules/multi_participant.py), such that,
```py
self.dataset['fit'] == concat_csv_data([load_csv(f'{participant}.csv').data for participant in train_participants])
self.dataset['val'] == concat_csv_data([load_csv(f'{participant}.csv').data for participant in val_participants])
self.dataset['test'] == concat_csv_data([load_csv(f'{participant}.csv').data for participant in test_participants])
```
Furthermore,
```py
if stage != 'test':
    # For normalisation
    train_mean, train_std = self.dataset['fit']['signal'].mean(), self.dataset['fit']['signal'].std()
```

3. The LSTM module is loaded, by making a combination between a [module that defines the metrics](https://github.com/AlexAntonides/stress-in-action/blob/main/src/__init__.py#L42) (i.e. BinaryAccuracy,F1Score,Precision,AUROC,etc.) and the [LSTM definition Itself](https://github.com/AlexAntonides/stress-in-action/blob/main/src/models/rnn.py) (in here you can find the layers), such that,
```md
[LSTM (hidden=128)] -> [Dropout (0.5)] -> [Linear(128, 1)]
```

4. For each step, the [windowed dataset](https://github.com/AlexAntonides/stress-in-action/blob/main/src/datasets/windowed.py) is used to window the data, such that,
```py
for window_size in self.dataset:
    if exceeds_length:
        continue
    if multiple_labels_present_in_data:
        continue
    X = window['signal']    # [amplitude_1, ..., amplitude_window_size]
    y = window['label'][0]  # 1 ("stressed")

    if stage != 'test':
        X = (x - self.train_mean) / self.train_std
    
    yield x, y
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-shield]: https://img.shields.io/badge/3.9%2B-yellow?style=for-the-badge&logo=python&logoColor=white&label=python&labelColor=blue
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/2.1%2B-orange?style=for-the-badge&logo=pytorch&logoColor=orange&label=PyTorch&labelColor=white
[pytorch-url]: https://pytorch.org/