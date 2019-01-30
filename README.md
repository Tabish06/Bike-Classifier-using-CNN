# Bike Classification Problem
Deep Learning Classification problem using Convolutional Neural Network.

This Github links helps in classifying image as a mountain bike or a road bike.

## TensorBoard Graph
![TensorBoard Graph](https://1drv.ms/u/s!Ahzgp0Lh4mOip0D52Vtk512c0nqI) 

# Setup
The Python 3 Anaconda Distribution is the easiest way to get going with the notebooks and code presented here.

(Optional) You may want to create a virtual environment for this repository:

```
conda create -n cv python=3 
source activate cv
```

This repository requires the installation of a few extra packages, you can install them all at once with:
```
pip install -r requirements.txt
```

# Training the model
```
python train.py

```
# Testing the model
All Images should be jpg format
## Testing from the folder
```
python test.py
```
## Testing from a filepath
```
python test.py -i file_path.jpg
```

Returns an ouput whether its a road bike or a mountain bike



