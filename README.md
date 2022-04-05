# Deep Learning Mini Project 1
Mini Project for ECE-GY 7123 Deep Learning (Spring '22) under Dr. Siddharth Garg in New York University.
* Aradhya Alamuru (aa9405@nyu.edu)
* Umesh Deshmukh (urd7172@nyu.edu)
* Mohammad Shamoon (ms12736@nyu.edu)

## Run the Project
* To reproduce our trained model architecture, ResNet on CIFAR-10 dataset
```python

model_path = './project1_model.pt'
net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
```
* To test the model on testdataset of CIFAR-10, in which we are able achieve accuracy of 90.6% 
run `self_eval.py`
```python
python self_eval.py

```

## Issues resolved:
- We have removed unnnecessary libraries, which are used later in project
- Rename `project1_model.py` (previous edition) to `resnet.py`
- Added `project1_model.py` contain model architecture which is needed to run `self_eval.py`
- Line 257 in `resnet.py` changed  to:
```python
torch.save(net.module.state_dict(), model_path)
```
- Added `project1_model.pt` trained weights

