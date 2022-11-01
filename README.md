# Harmonic Networks: Deep Translation and Rotation Equivariance

I implement the hard-baked locally rotationally equivariant convolutional neural networks (harmonic nets) in PyTorch found: https://arxiv.org/abs/1612.04642.

My equivariant function is based on learnable polynomials, however more features are to come. Loss metrics for deeper networks are quite promising, and show themselves in many applications that don't just suffer from translational equivariance, but further benefit from learnable local rotational equivariances.

Why'd this a good equivariance to learn?

Say an image-classifier is classifying butterflies from their image mid-flight. With the wing of the butterfly rotated in some way, the traditional CNN won't be able to learn from the rotation or consider butterflies with rotated wings as data to be equivariantly passed along - of course unless each frame example is given in the training data. Usually, only translations of said butterfly in the X and Y can be captured successfully. Hard-baking rotational equivariances in this example allows for wing rotation data to not be erased - therefore allowing for more complex feature compositions to be remembered and passed to the more general feature extractors that are dense nets.
