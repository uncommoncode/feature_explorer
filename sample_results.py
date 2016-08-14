import random
import model
import cPickle as pickle
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=model.CLASSIFIERS.keys(), default=model.DEFAULT_CLASSIFIER,
                        help="The classifier used for creating the probabilities")
    parser.add_argument("--sample", default=30, help="number of random samples to show")
    parser.add_argument("--top_k", default=5, help="top-k confident labels to show")
    parser.add_argument("--min_p", default=0.4, help="minimum confidence to allow")
    parser.add_argument("input_probs", help="The probabilities pickle created by run.py")
    args = parser.parse_args()

    classifier_ctor = model.CLASSIFIERS[args.classifier]
    classifier = classifier_ctor()
    with open(args.input_probs) as r:
        data = pickle.load(r)

    for image_path, scores in random.sample(data.items(), args.sample):
        # Filter based on min confidence.
        min_indices = np.argwhere(scores > args.min_p)
        # Filter the top confident results
        top_indices = scores.argsort()[::-1][:args.top_k]
        top_indices = top_indices[np.in1d(top_indices, min_indices)]
        top_labels = classifier.get_labels(top_indices)
        top_scores = scores[top_indices]
        # Print the score and label for each
        print image_path
        for score, label in zip(top_scores, top_labels):
            print "   %s: %s" % (score, label)

if __name__ == "__main__":
    main()
