import caffe
import numpy as np
import skimage
import os

CAFFE_ROOT = "/root/caffe/"
MODEL_ROOT = "/caffe"

IMAGENET_RESNET_152_MODEL = [
    os.path.join(MODEL_ROOT, "models/classifiers/imagenet/ResNet-152-deploy.prototxt"),
    os.path.join(MODEL_ROOT, "models/classifiers/imagenet/ResNet-152-model.caffemodel"),
    os.path.join(CAFFE_ROOT, "data/ilsvrc12/synset_words.txt")
]

PLACES_205_GOOGLENET_MODEL = [
    os.path.join(MODEL_ROOT, "models/classifiers/places205/deploy_places205.protxt"),
    os.path.join(MODEL_ROOT, "models/classifiers/places205/googlelet_places205_train_iter_2400000.caffemodel"),
    os.path.join(MODEL_ROOT, "models/classifiers/places205/categoryIndex_places205.csv")
]


def enable_gpu():
    caffe.set_mode_gpu()

def enable_cpu():
    caffe.set_mode_cpu()

def load_image(path):
    return caffe.io.load_image(path)


class PretrainedClassifier(object):
    def __init__(self):
        self.classifier = None
        self.labels = None
        self.feature_layer = None

    def _set_feature_layer_to_classification(self):
        self.feature_layer = self.classifier.blobs.keys()[-2]

    def predict_proba(self, image, oversample=True):
        return self.classifier.predict([image], oversample)[0]

    def extract_manifold_feature(self, image):
        # TODO(emmett): oversample here
        transformed_image = self.classifier.transformer.preprocess('data', image)
        self.classifier.blobs['data'].data[...] = transformed_image
        self.classifier.forward()
        return self.classifier.blobs[self.feature_layer].data

    def get_labels(self, indices):
        return self.labels[indices]


def load_imagenet_labels(labels_path):
    return np.loadtxt(labels_path, str, delimiter='\t')

class ImageNetClassifier(PretrainedClassifier):
    def __init__(self, model=IMAGENET_RESNET_152_MODEL, caffe_root=CAFFE_ROOT):
        pixelwise_means = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
        self.bgr_mean = pixelwise_means.mean(1).mean(1)
        model_path, weights_path, labels_path = model
        self.classifier = caffe.Classifier(model_path, weights_path,
                            mean=self.bgr_mean,
                            input_scale=None,
                            image_dims=(256, 256),
                            raw_scale=255.0,
                            channel_swap=(2, 1, 0))
        self.labels = load_imagenet_labels(labels_path)
        self._set_feature_layer_to_classification()


def load_places_205_labels(labels_path):
    return np.loadtxt(labels_path, str, delimiter=',')

class Places205Classifier(PretrainedClassifier):
    def __init__(self, model=PLACES_205_GOOGLENET_MODEL, caffe_root=CAFFE_ROOT):
        model_path, weights_path, labels_path = model
        # TODO(emmett): I don't know what the mean centering and other settings are yet.
        self.classifier = caffe.Classifier(model_path, weights_path)
        self.labels = load_places_205_labels(labels_path)
        self._set_feature_layer_to_classification()

CLASSIFIERS = {
    "imagenet": ImageNetClassifier,
    "places_205": Places205Classifier
}

DEFAULT_CLASSIFIER = "imagenet"
