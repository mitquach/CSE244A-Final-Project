# CSE244A-Final-Project

These directories include a selection of attempts to improve the performance of one of our
early models, testing which techniques seemed promising and which seemed to degrade performance.
In general, the results were mostly unhelpful.

gen_test_pred.py and fine_tune.py are two short tools I experimented with.  The first takes
an existing model and instead of training it, instead just generates predictions for the
entire training set.  The resulting file can be directly compared against the provided
train_labeled.csv file to see which images (and by extension which categories) the model
is having the most difficulty with.  The second takes an existing trained model and starts
a fresh training run using a different set of parameters.  Most of my experiments in this regard
were to take a model trained with a batch size of 16, and up the batch size to 64 (with the learning
rate reduced by a factor of 10 or 100).  The theory there was that the low batch size helped us
quickly converge toward a solution, but that in late-stage training it might be a liability because
the gradients of small batches would never really point toward the local minimum.  Using larger
batches at that point might encourage better convergence.  It didn't really seem to make a difference
to the outcomes though.
