# bayes

A Bayesian network. 

Note that this is a work in progress and that the only currently available functionality is that of the underlying naïve Bayes classifier.
The classifier is originally a translation from a Python example provided in an excellent blog post by Jace Kohlmeier at:

http://derandomized.com/post/20009997725/bayes-net-example-with-python-and-khanacademy

Apart from translating the example from Python + NumPy to C++ + Armadillo, this code expands on the example by providing functionality
for having an arbitrary amount of states for visible and hidden nodes/features.
