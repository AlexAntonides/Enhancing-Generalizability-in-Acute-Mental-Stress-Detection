# Stress-in-Action Library

## Overview
SiA is built on top of PyTorch and Lightning. 
The reason why the SiA library is made, is to allow these librarier to support large and distributed datasets, while also adding ECG-related preprocessing options and Stress models. As of writing, PyTorch allows you to support larger datasets through the use of an IterableDataset. However, it is a hassle to do so. 