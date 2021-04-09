from django.db import models
from tienda.models import Producto

class Orden(models.Model):
    nombre = models.CharField(max_length=50)
    primer_apellido = models.CharField(max_length=50)
    segundo_apellido = models.CharField(max_length=50)
    correo = models.EmailField()
    direccion = models.CharField(max_length=250)
    codigo_postal = models.CharField(max_length=20)
    ciudad = models.CharField(max_length=100)
    creado = models.DateTimeField(auto_now_add=True)
    actualizado = models.DateTimeField(auto_now=True)
    pagado = models.BooleanField(default=False)

    class Meta:
        ordering = ('-creado',)
        verbose_name = 'Orden'
        verbose_name_plural = 'Ordenes'

    def __str__(self):
        return f'Orden {self.id}'
    
    def obtener_costo_total(self):
        return sum(item.obtener_costo() for item in self.items.all())

class ItemOrden(models.Model):
    orden = models.ForeignKey(Orden, related_name='articulos',
                              on_delete=models.CASCADE)
    producto = models.ForeignKey(Producto, related_name='articulos_orden',
                                on_delete=models.CASCADE)
    precio = models.DecimalField(max_digits=10, decimal_places=2)
    cantidad = models.PositiveIntegerField(default=1)

    class Meta:
        verbose_name = 'Orden articulo'
        verbose_name_plural = 'Orden articulos'

    def __str__(self):
        return str(self.id)
    
    def obtener_costo(self):
        return self.precio * self.cantidad 