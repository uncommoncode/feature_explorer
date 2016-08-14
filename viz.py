import cPickle as pickle
import numpy as np
import projection
import landmark
import json

def save_viz_data_json(projection, landmarks, image_paths, output_path):
    landmark_labels = [image_paths[landmark_index] for landmark_index in landmarks]
    with open(output_path, "w") as w:
        points = []
        for i in xrange(len(projection)):
            point = projection[i, :]
            points.append({"img": image_paths[i], "x": point[0], "y": point[1]})
        json.dump({"points": points, "landmarks": landmark_labels}, w)

def load_probabilities(path):
    with open(path) as r:
        data = pickle.load(r)
    # RAM inefficient but functional unzip/concat.
    image_paths, values = zip(*data.items())
    if len(values[0].shape) == 1:
        # Convert to vector
        values = [np.reshape(value, (-1, len(value))) for value in values]
    points = np.concatenate(values)
    return image_paths, points

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", choices=projection.PROJECTIONS, default=projection.DEFAULT_PROJECTION,
                        help="The dimensionality reduction technique")
    parser.add_argument("--landmark", choices=landmark.LANDMARKS, default=landmark.DEFAULT_LANDMARK,
                        help="The landmark creation technique")
    parser.add_argument("--landmark_count", default=10, help="The number of landmarks to output")
    parser.add_argument("--output_path", default="viz/static/json/creatures_data.json",
                        help="The vizualization json file to output")
    parser.add_argument("input_probs", help="The probabilities pickle created by run.py")
    args = parser.parse_args()

    run_projection = projection.PROJECTIONS[args.projection]
    run_landmark = landmark.LANDMARKS[args.landmark]

    image_paths, points = load_probabilities(args.input_probs)

    projeted_points = run_projection(points)
    landmarks = run_landmark(projeted_points, args.landmark_count)
    save_viz_data_json(projeted_points, landmarks, image_paths, args.output_path)

if __name__ == "__main__":
    main()
