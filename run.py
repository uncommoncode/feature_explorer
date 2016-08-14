import util
import model
import transformation
import cPickle as pickle

def find_images(input_dir):
    images = []
    images += list(util.find_files(input_dir, "*.jpg"))
    images += list(util.find_files(input_dir, "*.JPG"))
    return images

def save_checkpoint(output_dir, name, value):
    with open("%s/%s.pkl" % (output_dir, name), "w") as w:
        pickle.dump(value, w)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=model.CLASSIFIERS.keys(), default=model.DEFAULT_CLASSIFIER,
                        help="The classifier to run.")
    parser.add_argument("--transformation", choices=transformation.TRANSFORMATIONS.keys(),
                        default=transformation.DEFAULT_TRANSFORMATION,
                        help="The transformation to apply to extract results from the classifier")
    parser.add_argument("--sample", type=int, default=None,
                        help="The number of images to run over. Leave out to run over all images.")
    parser.add_argument("--checkpoint_interval", default=5, help="The interval to checkpoint data for output.")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration. This needs large amount of GPU RAM for some cases.")
    parser.add_argument("input_dir", help="The directory of input images.")
    parser.add_argument("output_dir", help="The directory to output the probabilities pickle file used by viz.py and "
                                           "sample_results.py")
    args = parser.parse_args()

    if args.gpu:
        model.enable_gpu()
    else:
        model.enable_cpu()

    classifier_ctor = model.CLASSIFIERS[args.classifier]
    run_transformation = transformation.TRANSFORMATIONS[args.transformation]

    image_paths = util.stable_pseudo_shuffle(find_images(args.input_dir))
    if args.sample is not None:
        image_paths = image_paths[:args.sample]

    output_name = "%s_%s" % (args.classifier, args.transformation)
    for index, update in enumerate(run_transformation(classifier_ctor(), image_paths)):
        # Checkpoint the results every few updates.
        if index % args.checkpoint_interval == 0:
            save_checkpoint(args.output_dir, output_name, update)
    # Ensure the last update is written out.
    save_checkpoint(args.output_dir, output_name, update)

if __name__ == "__main__":
    main()
