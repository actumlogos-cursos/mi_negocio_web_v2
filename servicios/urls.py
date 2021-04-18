from django.urls import path
from . import views

app_name = 'servicios'

urlpatterns = [
    path('mnist-inicio', views.mnist_inicio, name='mnist'),
    path('jpg-to-pdf', views.jpg2pdf, name='jpg2pdf'),
    path('transferencia-estilo', views.transferencia_estilo, name='transferencia-estilo'),
]