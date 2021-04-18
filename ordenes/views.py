from django.shortcuts import render
from .models import ItemOrden
from .formulario import FormCreacionOrden
from carrito.carrito import Carrito

import stripe
from django.http import JsonResponse
from django.conf import settings

stripe.api_key = settings.STRIPE_SECRET_KEY

def crear_orden(request):
    carrito = Carrito(request)
    if request.method == 'POST':
        formulario = FormCreacionOrden(request.POST)
        if formulario.is_valid():
            orden = formulario.save()
            for item in carrito:
                ItemOrden.objects.create(orden=orden, producto=item['producto'],
                                         precio=item['precio'],
                                         cantidad=item['cantidad'])
            # Limpiar el carrito
            precio = carrito.obtener_precio_total
            carrito.limpiar_carrito()
            return render(request, 
                          'pagos/stripe.html', 
                          {'orden': orden,
                          'precio': precio,
                          'STRIPE_PUBLIC_KEY': settings.STRIPE_PUBLIC_KEY})
    else:
        formulario = FormCreacionOrden()
    return render(request, 
                  'ordenes/crear.html', 
                  {'carrito': carrito, 'formulario': formulario})

def pago_exitoso(request):
    return render(request, 'ordenes/creado.html')

def pago_cancelado(request):
    return render(request, 'ordenes/cancelado.html')

def SesionPagoStripe(request, precio):
    YOUR_DOMAIN = "http://127.0.0.1:8000"
    precio= int(float(precio))*100
    checkout_session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[
            {
                'price_data': {
                    'currency': 'mxn',
                    'unit_amount': precio,
                    'product_data': {
                        'name': 'Tu pedido',
                        'images': ['https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg'],
                    },
                },
                'quantity': 1,
            },
        ],
        mode='payment',
        success_url=YOUR_DOMAIN + '/ordenes/success/',
        cancel_url=YOUR_DOMAIN + '/ordenes/cancel/',
    )
    return JsonResponse({
        'id': checkout_session.id
    })