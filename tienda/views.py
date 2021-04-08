from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import Categoria, Producto

# Create your views here.
def tienda_inicio(request):
    return HttpResponse("<h1> Bienvenido a la tienda </h1>")

def lista_productos(request, categoria_slug=None):
    categoria = None
    categorias = Categoria.objects.all()
    print('Las categorias son: ')
    print(categorias)
    productos = Producto.objects.filter(disponibilidad=True)
    print('Los productos son: ')
    print(productos)
    if categoria_slug:
        categoria = get_object_or_404(Categoria, slug=categoria_slug)
        productos = productos.filter(categoria=categoria)
    print('Llegamos aqui')
    return render(request, 'tienda/productos/lista.html',
                    {'categoria': categoria,
                    'categorias': categorias,
                    'productos': productos})

from carrito.formulario import FormularioAgregarProducto

def producto_detalle(request, id, slug):
    producto = get_object_or_404(Producto, id=id,
                                slug=slug, disponibilidad=True)
    form_carrito_producto = FormularioAgregarProducto()
    return render(request, 'tienda/productos/detalles.html', 
                  {'producto': producto,
                   'form_carrito_producto':form_carrito_producto})
