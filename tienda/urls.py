from django.urls import path
from . import views

app_name = 'tienda'

urlpatterns = [
    path('', views.lista_productos, name='lista_productos'),
    path('<slug:categoria_slug>/', views.lista_productos,
        name='lista_productos_por_categoria'),
    path('<int:id>/<slug:slug>/', views.producto_detalle, 
        name='producto_detalle'),
]