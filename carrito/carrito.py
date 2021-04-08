from decimal import Decimal
from django.conf import settings
from tienda.models import Producto

class Carrito(object):

    def __init__(self, request):
        """
        Inicializacion del carrito
        """
        self.session = request.session
        carrito = self.session.get(settings.CARRITO_SESSION_ID)
        if not carrito:
            # Guardar un carrito vacio en la sesion
            carrito = self.session[settings.CARRITO_SESSION_ID] = {}
        self.carrito = carrito
    
    def add(self, producto, cantidad=1, cambiar_cantidad=False):
        """
        Añade un producto al carrito o actualiza su cantidad
        """
        producto_id = str(producto.id)
        if producto_id not in self.carrito:
            self.carrito[producto_id] = {'cantidad': 0,
                                         'precio': str(producto.precio)}
        if cambiar_cantidad:
            self.carrito[producto_id]['cantidad'] = cantidad
        else:
            self.carrito[producto_id]['cantidad'] += cantidad
        self.guardar()
    
    def guardar(self):
        self.session.modified = True
    
    def remover(self, producto):
        """
        Remueve un producto del carrito
        """
        producto_id = str(producto.id)
        if producto_id in self.carrito:
            del self.carrito[producto_id]
            self.guardar()
    
    def __iter__(self):
        """
        iterar sobre los articulos en el carrito y
        obtenerlos de la base de datos
        """
        productos_ids = self.carrito.keys()
        productos = Producto.objects.filter(id__in=productos_ids)
        carrito = self.carrito.copy()
        for producto in productos:
            carrito[str(producto.id)]['producto'] = producto
        for item in carrito.values():
            item['precio'] = Decimal(item['precio'])
            item['precio_total'] = item['precio']*item['cantidad']
            yield item
    
    def __len__(self):
        """
        Conteo de productos en el carrito
        """
        return sum(item['cantidad'] for item in self.carrito.values())
    
    def obtener_precio_total(self):
        return sum(Decimal(item['precio']) * item['cantidad'] 
                    for item in self.carrito.values())
    
    def limpiar_carrito(self):
        #Elimina el carrito de la sesion
        del self.session[settings.CARRITO_SESSION_ID]
        self.guardar()

