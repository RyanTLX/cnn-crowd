from django.shortcuts import render
from django.http import JsonResponse
import os
import random
import time

dataset_path = 'CrowdEstimatorWeb/static/CrowdEstimatorWeb/CrowdImages'
files = [f for f in os.listdir(dataset_path) if not f.startswith('.')]

def reload_data(request):
    params = []

    # Get 6 random images from CrowdDataset folder.
    for i in range(6):
        image_name = files[random.randint(1, len(files))]
        label = image_name.split('_')[0] # CHANGE TO PREDICTION
        color = "success"
        if label == "medium":
            color = "warning"
        elif label == "high":
            color = "danger"
        params.append({'cabin_image_name': image_name, 'cabin_label': label, 'cabin_color': color})

    return JsonResponse({'params': params})

def index(request):
    return render(request, 'CrowdEstimatorWeb/index.html')
