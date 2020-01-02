Snn_simple is a spiking neual network implementation written for educational purposes alone. It was
designed to be a simple as possible, using fewer than 150 lines of python with numpy as the only
library.

This SNN uses only exitatory neurons and a very basic implementation of the hebbian learning 
principle. The "dataset" it learns from consists of only two examples, one from each category.
There is no train/test split, the purpose of the model is only to demonstrate rudamentary hebbian
learning in a spiking neural network by fitting to the given vectors.

Please note than the network may not reliably converge. If fact, it's highly likely it will get 
perfect accuracy in one training iteration and <30% accuracy in the next. The implementation could
be stabilized considerably with the addition of anti-hebbian behavior or inhibitory neurons.
