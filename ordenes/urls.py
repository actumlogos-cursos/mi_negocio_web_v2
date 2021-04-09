from django.urls import path
from . import views

app_name = 'ordenes'

urlpatterns = [
    path('crear/', views.crear_orden, name='crear_orden'),
]
