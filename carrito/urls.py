from django.urls import path
from . import views

app_name = 'carrito'

urlpatterns = [
    path('', views.carrito_detalle, name='carrito_detalle'),
    path('add/<int:producto_id>/', views.agregar_carrito, name='agregar_carrito'),
    path('remover/<int:producto_id>/', views.remover_carrito, name='remover_carrito'),
]