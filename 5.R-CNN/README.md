# R-CNN

##
Object Detection의 논문들을 읽다보면 R-CNN을 기반으로 하는 논문들이 많다. 때문에 R-CNN으 꼭 읽어보아야 하느 논문이며 구현해봐야 하는 논문이라고 생각한다. 하지만 R-CNN 구현을 위해서는 매우 복잡한 코드를 작성해야한다. 때문에 필자는 Github에서 zjZSTU님이 구현한 R-CNN코드[[2](https://github.com/object-detection-algorithm/R-CNN/tree/a7a66144a1809cde728052b4c392ef7d92fdab97)]를 이용하여 필자 입맛에 맞게 수정하였다. 자세한 코드 설명은 [링크](https://ctkim.tistory.com/191)를 통해서 확인할 수 있다.


## Requirements
* python3
* PyTorch
* torchvision 
* numpy 


## DataSet
* [PASCAL DATASET](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)


## Training & Evaluating





# References
[[1](https://arxiv.org/pdf/1409.1556.pdf)] Karen Simonyan, Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition", 	arXiv:1409.1556.   
[[2](https://github.com/object-detection-algorithm/R-CNN/tree/a7a66144a1809cde728052b4c392ef7d92fdab97)] R-CNN Pytorch Github
