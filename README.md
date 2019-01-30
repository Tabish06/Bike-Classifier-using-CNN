# Bike Classification Problem
Deep Learning Classification problem using Convolutional Neural Network.

This Github links helps in classifying image as a mountain bike or a road bike.
Add data to data folder divided in mountain bike and road bikes for training

## TensorBoard Graph
![TensorBoard Graph](https://github.com/Tabish06/Bike-Classifier-using-CNN/blob/master/images/2019-01-29%20(2).png) 

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
This should be used for evaluating images for testing. So the images should be classified to calculate the testing accuracy

```
python test.py
```
## Testing from a filepath
```
python test.py -i file_path.jpg
```

Returns an ouput whether its a road bike or a mountain bike

# Output
![Output](https://github.com/Tabish06/Bike-Classifier-using-CNN/blob/master/images/2019-01-29%20(3).png)

![Output](https://github.com/Tabish06/Bike-Classifier-using-CNN/blob/master/images/f6cda5cbba9f5ab71fee3ce6eec5e3b0.png)
