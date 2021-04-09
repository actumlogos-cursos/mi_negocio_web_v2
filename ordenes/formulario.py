from django import forms
from .models import Orden

class FormCreacionOrden(forms.ModelForm):
    class Meta:
        model = Orden
        fields = ['nombre', 'primer_apellido', 'segundo_apellido', 'correo',
        'direccion', 'codigo_postal', 'ciudad']
