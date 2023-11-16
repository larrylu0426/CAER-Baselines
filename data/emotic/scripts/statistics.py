import ast
import os
import pathlib

import pandas as pd

PROJECT_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent

phases = ['train', 'val', 'test']
cat_labels = ['Affection', 'Anger', 'Annoyance', 'Anticipation', \
                'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
                'Disquietment', 'Doubt/Confusion', 'Embarrassment', \
                'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',\
                'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', \
                'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

for phase in phases:
    n = 0
    counts = {}
    for emotion in cat_labels:
        counts[emotion] = 0
    data = pd.read_csv(os.path.join(PROJECT_DIR, '{}.csv'.format(phase)))
    for index, row in data.iterrows():
        labels = ast.literal_eval(row['Categorical_Labels'])
        for label in labels:
            counts[label] += 1
        n += 1
    print("Phase: {}".format(phase))
    print("Total: {}".format(n))
    print("Counts: \n {}".format(counts))
