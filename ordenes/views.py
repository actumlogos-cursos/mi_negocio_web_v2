from django.shortcuts import render
from .models import ItemOrden
from .formulario import FormCreacionOrden
from carrito.carrito import Carrito

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
            carrito.limpiar_carrito()
            return render(request, 
                          'ordenes/creado.html', 
                          {'orden': orden})
    else:
        formulario = FormCreacionOrden()
    return render(request, 
                  'ordenes/crear.html', 
                  {'carrito': carrito, 'formulario': formulario})