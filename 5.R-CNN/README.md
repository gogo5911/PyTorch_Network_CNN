# VGG


## Requirements
* python3
* PyTorch
* torchvision 
* numpy 


## DataSet
* [STL10 dataset](https://cs.stanford.edu/~acoates/stl10/)


## Training & Evaluating


```
optional arguments:
 -- type (default : 'train')
 -- resume (default : -1) 
 -- epoch (default : 15)
 -- path 
```


```
python main.py

python main.py -t train

python main.py -t -e 30

python main.py -t train -r 8 

python main.py -t traon -r 8 -e 30

python main.py -t test -p ./checkpoint/epoch0.pth
```


# References
[[1](https://arxiv.org/pdf/1409.1556.pdf)] Karen Simonyan, Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition", 	arXiv:1409.1556
