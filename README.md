# feature_explorer
Deep Neural Network feature exploration tool providing insights into the feature space for an image corpus. 

For example, this can help to justify the cost of labeling data. If the features provide relavant similarity projected in two dimensions, then the higher dimensional space should be useful for fine-tuning or training a new classifier with better labeled data.

# Usage:
 * run.py: take a pretrained classifier and output probabilities for a directory of jpg files
 * viz.py: take output probabilities and write to a json used for visualization with `viz/static/d3_viz.htm`
 * sample_results.py: print out example classifications above some confidence threshold for a sample of images in a directory

# Requirements:
 * Caffe
 * [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) installed to `/caffe/models/classifiers/imagenet/`
 * [Places205-GoogLeNet](http://places.csail.mit.edu/model/googlenet_places205.tar.gz) installed to `/caffe/models/classifiers/places205/`
