# AlexNet


## Requirements
* python3
* PyTorch
* torchvision 
* numpy 


## DataSet
* [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)


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
[[1](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)] Krizhevsky, Alex; Sutskever, Ilya; Hinton, Geoffrey E. "ImageNet classification with deep convolutional neural networks". Communications of the ACM. 60 (6): 84–90. doi:10.1145/3065386. ISSN 0001-0782. 
