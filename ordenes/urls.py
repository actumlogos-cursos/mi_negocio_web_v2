from django.urls import path
from . import views

app_name = 'ordenes'

urlpatterns = [
    path('crear/', views.crear_orden, name='crear_orden'),
    path('pagar-pedido-stripe/<precio>', views.SesionPagoStripe, name='pago-stripe'),
    path('cancel/', views.pago_cancelado, name='cancelado'),
    path('success/', views.pago_exitoso, name='exitoso'),
    path('pagar-pedido-paypal', views.pago_paypal, name='pago-paypal'),
]
