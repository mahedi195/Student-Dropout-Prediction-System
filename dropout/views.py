from django.shortcuts import render
import numpy as np

from .ml_model import model, scaler, imputer


def home(request):
    return render(request, 'home.html')

def predict(request):
    context = {
        "n1": request.GET.get("n1", ""),
        "n2": request.GET.get("n2", ""),
        "n3": request.GET.get("n3", "0"),
        "n4": request.GET.get("n4", ""),
        "n5": request.GET.get("n5", ""),
        "n6": request.GET.get("n6", ""),
        "n7": request.GET.get("n7", ""),
    }
    return render(request, 'predict.html', context)

def result(request):
    values = np.array([
        float(request.GET.get('n1', 0)),
        float(request.GET.get('n2', 0)),
        float(request.GET.get('n3', 0)),
        float(request.GET.get('n4', 0)),
        float(request.GET.get('n5', 0)),
        float(request.GET.get('n6', 0)),
        float(request.GET.get('n7', 0)),
    ]).reshape(1, -1)

    values = imputer.transform(values)
    values = scaler.transform(values)

    pred = model.predict(values)

    result1 = "will DROP OUT" if pred[0][0] > 0.5 else "won't DROP OUT"

    return render(request, "predict.html", {
        "result2": result1,
        "n1": request.GET.get("n1", ""),
        "n2": request.GET.get("n2", ""),
        "n3": request.GET.get("n3", ""),
        "n4": request.GET.get("n4", ""),
        "n5": request.GET.get("n5", ""),
        "n6": request.GET.get("n6", ""),    
        "n7": request.GET.get("n7", ""),
    }) 