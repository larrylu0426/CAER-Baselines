import ast
import os
import pathlib

PROJECT_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent

phases = ['train', 'test']
cat_labels = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
]

for phase in phases:
    n = 0
    counts = {}
    for emotion in cat_labels:
        counts[emotion] = 0
    data = [
        line.removesuffix("\n")
        for line in open(os.path.join(PROJECT_DIR, '{}.txt'.format(phase)),
                         'r').readlines()
    ]
    for item in data:
        sample = item.split(',')
        label = ast.literal_eval(sample[1])
        counts[cat_labels[label]] += 1
        n += 1
    print("Phase: {}".format(phase))
    print("Total: {}".format(n))
    print("Counts: \n {}".format(counts))
