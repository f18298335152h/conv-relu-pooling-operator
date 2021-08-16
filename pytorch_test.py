import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def test_relu():
    inputs = np.array([
        0.0, 1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        2.0, 3.0, 4.0, 5.0, 6.0,
        3.0, 4.0, 5.0, 6.0, 7.0,
        4.0, 5.0, 6.0, 7.0, 8.0])
    input_tensor = torch.from_numpy(inputs)
    output = F.relu(input_tensor)
    print("relu output = ", output)


def test_pooling():
    inputs = np.array([
        0.0, 1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        2.0, 3.0, 4.0, 5.0, 6.0,
        3.0, 4.0, 5.0, 6.0, 7.0,
        4.0, 5.0, 6.0, 7.0, 8.0]).astype(np.float32) 

   
    inputs = np.reshape(inputs,(5,5,1)) 
    input_tensor = torch.from_numpy(inputs)
    output = F.max_pool2d(input_tensor, kernel_size=(2,2),padding=1)
    print("pooling output = ", output)


def test_convolution():
    w=torch.nn.Conv2d(1,1,3,stride=(2,2),padding=0)
    weight_np = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]]).astype(np.float32)

    weight_np = np.expand_dims(weight_np, axis=0);
    weight_np = np.expand_dims(weight_np, axis=0);
    weight_tensor = torch.from_numpy(weight_np);
    w.weight = torch.nn.Parameter(weight_tensor);

    inputs = np.array([
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0, 8.0]]).astype(np.float32)
    
    input_np = np.expand_dims(inputs, axis=0);
    input_np = np.expand_dims(input_np, axis=0);
    input_tensor = torch.from_numpy(input_np)

    output = w(input_tensor)
    print("conv output = ", output)



if __name__=="__main__":
    test_relu()
    test_pooling()
    test_convolution()
