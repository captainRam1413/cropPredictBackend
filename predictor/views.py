from rest_framework.decorators import api_view
from rest_framework.response import Response
from .ml_logic import predict_crop_fertilizer, get_gemini_advice
from django.http import HttpResponse

@api_view(['POST'])
def predict_view(request):
    required_fields = ['district', 'soil', 'nitrogen', 'potassium', 'phosphorus', 'ph', 'rainfall', 'temperature']
    for field in required_fields:
        if field not in request.data:
            return Response({'error': f'Missing field: {field}'}, status=400)

    result = predict_crop_fertilizer(request.data)

    return Response(result)

@api_view(['POST'])
def advice_view(request):
    required_fields = ['district', 'soil', 'nitrogen', 'potassium', 'phosphorus', 'ph', 'rainfall', 'temperature', 'predicted_crop', 'predicted_fertilizer']
    for field in required_fields:
        if field not in request.data:
            return Response({'error': f'Missing field: {field}'}, status=400)

    result = get_gemini_advice(request.data)

    if 'error' in result:
        return Response(result, status=400)

    return Response({'advice': result})

def ping(request):
    return HttpResponse("pong")


