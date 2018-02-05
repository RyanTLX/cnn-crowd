from django.shortcuts import render
import os
import random

# Create your views here.
def index(request):
    params = []

    # Get 6 random images from CrowdDataset folder.
    dataset_path = 'CrowdEstimatorWeb/static/CrowdEstimatorWeb/CrowdImages'
    files = [f for f in os.listdir(dataset_path) if not f.startswith('.')]

    for i in range(6):
        image_name = files[random.randint(1, len(files))]
        label = image_name.split('_')[0] # CHANGE TO PREDICTION
        color = "success"
        if label == "medium":
            color = "warning"
        elif label == "high":
            color = "danger"
        params.append({'image_name': image_name, 'label': label, 'color': color})

    return render(request, 'CrowdEstimatorWeb/index.html', {'params': params})
