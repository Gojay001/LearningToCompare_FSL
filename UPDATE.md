# LearningToCompare_FSL
PyTorch code for CVPR 2018 paper: [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025) (Few-Shot Learning part). 

Upadated in some files to fit new PyTorch version and personal task.  

## Original Files
---
```
|- datas
    |- miniImagenet
    |- omniglot_resize.zip
|- LICENSE
|- miniimagenet
    |- miniimagenet_test_few_shot.py
    |- miniimagenet_test_one_shot.py
    |- miniimagenet_train_few_shot.py
    |- miniimagenet_train_one_shot.py
    |- models
    |- task_generator_test.py
    |- task_generator.py
|- omniglot
    |- models
    |- omniglot_test_few_shot.py
    |- omniglot_test_one_shot.py
    |- omniglot_train_few_shot.py
    |- omniglot_train_one_shot.py
    |- task_generator.py
|- README.md
```

> **datas** : Follows the `README.md` to download the miniImagenet and omniglot resource data.  
> **miniimagenet** : Includes preprocess, train, test and model files of miniimagenet dataset.  
> **omniglot** : Includes preprocess, train, test and model files of omniglot dataset.  

It modified some problems for running only, more details will be written in later part.  


## Additional Files
---
```
|- cpu_gpu
    |- miniimagenet
    |- omniglot
```

### Features
1. solve the diffrent version of PyTorch. `Original version` : python2.7 + pytorch0.4, `New version` : python3.6 + pytorch1.0.  
2. update the file for running in both cpu and gpu available.  

### Details

#### output with shape [1, 28, 28] doesn’t match the broadcast shape [3, 28, 28]
- **Why** :   
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) will cause below error: 
`RuntimeError: output with shape [1, 28, 28] doesn’t match the broadcast shape [3, 28, 28]`
- **Way** :  
change code as follows in *'get_data_loader'* of *'task_generator.py'* :  
```python
- normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
+ normalize = transforms.Normalize(mean=[0.5], std=[0.5])
```
 
#### Expected object of scalar type Long but got scalar type Int for argument #3 ‘index’
- **Why** :  
build-in element type are Int, but expected Long type here.
- **Way** :  
change code as follows in *'omniglot_train_one_shot.py'* :
```python
mse = nn.MSELoss().cuda(GPU)
- one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, 1, batch_labels.view(-1,1))).cuda(GPU)
+ one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1,  batch_labels.view(-1, 1).long(),1)).cuda(GPU)
loss = mse(relations,one_hot_labels)
```

#### RuntimeError: Expected object of backend CUDA but got backend CPU for argument #2 ‘other’
- **Why** :  
```
rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM)]
```

print the value of `predict_label` and `test_labels` :  
```
predict_labels[j] =tensor([4, 0, 3, 1, 2], device='cuda:0')
test_labels[j] = tensor([4, 0, 3, 1, 2], dtype=torch.int32)
```
- **Solve** :  
change code as follows in *'omniglot_train_one_shot.py'* :  
```python
+ predict_labels = predict_labels.to(device, dtype=torch.int32)
+ test_labels = test_labels.to(device, dtype=torch.int32)
```

#### IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python
- **Why** :  
```
 train_loss += loss.data[0]
```
this code cause error due to PyTorch version changed.  
- **Solve** :  
change code as follows in *'omniglot_train_one_shot.py'* :  
```python
- train_loss += loss.data[0]
+ train_loss += loss.item()
```

## Task Files
---
```
|- fiber
    |- datas
    |- fiber_test_few_shot.py
    |- fiber_test_one_shot.py
    |- fiber_train_few_shot.py
    |- fiber_train_one_shot.py
    |- models
    |- task_generator_test.py
    |- task_generator.py
    |- test.py
```

> It includes own train and test datasets to process task.