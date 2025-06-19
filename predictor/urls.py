from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_view),
    path('ping/', views.ping, name='ping'),
]
