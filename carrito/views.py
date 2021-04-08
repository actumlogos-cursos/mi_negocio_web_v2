from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST
from tienda.models import Producto
from .carrito import Carrito
from .formulario import FormularioAgregarProducto

@require_POST
def agregar_carrito(request, producto_id):
    carrito = Carrito(request)
    producto = get_object_or_404(Producto, id=producto_id)
    formulario = FormularioAgregarProducto(request.POST)
    if formulario.is_valid():
        cd = formulario.cleaned_data
        carrito.add(producto=producto, cantidad=cd['cantidad'],
                 cambiar_cantidad=cd['sobreescribir'])
    return redirect('carrito:carrito_detalle')

@require_POST
def remover_carrito(request, producto_id):
    carrito = Carrito(request)
    producto = get_object_or_404(Producto, id=producto_id)
    carrito.remover(producto)
    return redirect('carrito:carrito_detalle')

def carrito_detalle(request):
    carrito = Carrito(request)
    for item in carrito:
        item['form_actual_cantidad'] = FormularioAgregarProducto(initial={
            'cantidad': item['cantidad'],
            'sobreescribir': True})
    return render(request, 'carrito/detalle.html', {'carrito': carrito})