from django.urls import path
from . import views

app_name = 'servicios'

urlpatterns = [
    path('mnist-inicio', views.mnist_inicio, name='mnist'),
]