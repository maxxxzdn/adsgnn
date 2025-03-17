# AdS-GNN - a Conformally Equivariant Graph Neural Network

<p align="center">
    <a href="https://arxiv.org/abs/2502.17019"><strong>AdS-GNN - a Conformally Equivariant Graph Neural Network</strong></a><br/>
    <a href="https://maxxxzdn.github.io/">Max Zhdanov</a>, Nabil Iqbal, Erik Bekkers, Patrick Forrré
</p>

### Abstract
Conformal symmetries, i.e. coordinate transformations that preserve angles, play a key role in many fields, including physics, mathematics, computer vision and (geometric) machine learning. Here we build a neural network that is equivariant under general conformal transformations. To achieve this, we lift data from flat Euclidean space to Anti de Sitter (AdS) space. This allows us to exploit a known correspondence between conformal transformations of flat space and isometric transformations on the Anti de Sitter space. We then build upon the fact that such isometric transformations have been extensively studied on general geometries in the geometric deep learning literature. In particular, we then employ message-passing layers conditioned on the proper distance, yielding a computationally efficient framework. We validate our model on point cloud classification (SuperPixel MNIST) and semantic segmentation (PascalVOC-SP).

### Requirements:
- jax
- flax

to validate installation, run
```python
python adsgnn.py
```
which should return the output shape (10, 64).