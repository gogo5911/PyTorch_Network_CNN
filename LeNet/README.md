# LeNet-5


## Requirements
* python3
* PyTorch
* torchvision 
* numpy 


## DataSet
* [MNIST dataset](http://yann.lecun.com/exdb/mnist/)


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
[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
