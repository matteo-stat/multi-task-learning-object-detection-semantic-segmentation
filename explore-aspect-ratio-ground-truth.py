import json
import csv
import matplotlib.pyplot as plt
import numpy as np

# list to store aspect ratios
aspect_ratios = {'monorail': [], 'person': [], 'forklift': []}

# labels conversions
label_code_to_str = {
    1: 'monorail',
    2: 'person',
    3: 'forklift'
}

# files to load
filenames = ['train', 'eval', 'test']

# for each file
for filename in filenames:

    # read training data
    with open(f'data/{filename}.json', 'r') as f:
        data = json.load(f)

    # sample training data
    for _, _, path_labels_boxes in data:

        # read labels boxes
        with open(path_labels_boxes, 'r') as f:
            labels_boxes = list(csv.reader(f))

        # for each box
        for label, xmin, ymin, xmax, ymax in labels_boxes:

            # data conversion
            label = int(label)
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)
            width = xmax - xmin + 1.0
            height = ymax - ymin + 1.0

            # append aspect ratio
            aspect_ratios[label_code_to_str[label]].append(width / height)

# create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# plot histograms for each class
colors = ['blue', 'green', 'orange']

for idx, (class_name, ratios) in enumerate(aspect_ratios.items()):
    ratios = np.array(ratios)
    axs[idx].hist(ratios, bins=20, alpha=0.7, color=colors[idx], weights=np.zeros_like(ratios) + 1. / ratios.size)
    axs[idx].set_title(f'aspect ratio distribution for {class_name}')
    axs[idx].grid(True)
    axs[idx].set_ylabel('relative frequency')

plt.xlabel('aspect ratio')
fig.legend(aspect_ratios.keys(), loc='upper right')
plt.tight_layout()
plt.show()

for key, value in aspect_ratios.items():

    # measures of central tendency
    mean = np.mean(value)
    median = np.median(value)
    q1 = np.percentile(value, 25)
    q3 = np.percentile(value, 75)
    
    # measures of dispersion
    min = np.amin(value)
    max = np.amax(value)
    range = np.ptp(value)
    variance = np.var(value)
    sd = np.std(value)

    print(f'\n------- {key} -------')
    print(f'mean: {mean:.4f}')
    print(f'q1: {q1:.4f}')
    print(f'median: {median:.4f}')
    print(f'q3: {q3:.4f}')
    print(f'minimum: {min:.4f}')
    print(f'maximum: {max:.4f}')
    print(f'range: {range:.4f}')
    print(f'variance: {variance:.4f}')
    print(f'standard deviation: {sd:.4f}')