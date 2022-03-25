# Deep Learning Mini Project 1
Mini Project for ECE-GY 7123 Deep Learning (Spring '22) under Dr. Siddharth Garg in New York University.
* Aradhya Alamuru (aa9405@nyu.edu)
* Umesh Deshmukh (urd7172@nyu.edu)
* Mohammad Shamoon (ms12736@nyu.edu)

## Run the Project
* To reproduce our trained model architecture, ResNet on CIFAR-10 dataset
```
model_path = './project1_model.pt'
net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
```
* To test the model
``` 
from Project_Model_1 import testing_best_model
testing_best_model()
```