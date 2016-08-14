import model

def run_probabilities(classifier, image_paths):
    results = {}
    for path in image_paths:
        print "Processing: %s" % path
        image = model.load_image(path)
        results[path] = classifier.predict_proba(image)
        yield results

def run_features(classifier, image_paths):
    results = {}
    for path in image_paths:
        print "Processing: %s" % path
        image = model.load_image(path)
        results[path] = classifier.extract_manifold_feature(image)
        yield results

TRANSFORMATIONS = {
    "probabilities": run_probabilities,
    "features": run_features
}
