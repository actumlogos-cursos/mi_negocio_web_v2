from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings
import tensorflow as tf
import json

import img2pdf
from django.core.files.uploadedfile import InMemoryUploadedFile
import io


def mnist_inicio(request):
    if request.method == "POST":
        model_graph = tf.compat.v1.Graph()
        with model_graph.as_default():
            tf_session = tf.compat.v1.Session()
            with tf_session.as_default():
                model=load_model('mi_negocio_web/static/ml/modelo_digitos.h5')
        img_height, img_width=28,28
        fileObj=request.FILES['files']
        fs=FileSystemStorage()
        archivo=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(archivo)
        testimage='.'+filePathName
        img = image.load_img(testimage, color_mode="grayscale", 
                            target_size=(img_height, img_width))

        x = image.img_to_array(img)
        x = x/255
        x = x.reshape(1,img_height, img_width, 1)
        with model_graph.as_default():
            with tf_session.as_default():
                predi=model.predict_classes(x)
        predictedLabel = predi[0]
        contexto = {'url':filePathName,
                    'predictedLabel':predictedLabel}
        return render(request,'servicios/mnist.html', contexto)
    return render(request, 'servicios/mnist.html')

def jpg2pdf(request):
    if request.method == "POST":
        fileObj = request.FILES['files']
        a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
        layout_fun = img2pdf.get_layout_fun(a4inpt)
        archivo = img2pdf.convert(fileObj, layout_fun=layout_fun)
        data = InMemoryUploadedFile(
            file=io.BytesIO(archivo),
            field_name='jpg2pdf',
            name='{}.pdf'.format('jpg2pdf'),
            content_type='pdf',
            size=len(archivo),
            charset='utf-8',
        )
        fs=FileSystemStorage()
        archivo=fs.save(data.name, data)
        filePathName=fs.url(archivo)
        contexto = {'url': filePathName}
        return render(request, 'servicios/jpg2pdf.html', contexto)
    return render(request, 'servicios/jpg2pdf.html')
