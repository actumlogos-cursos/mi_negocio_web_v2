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

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg16
from scipy.optimize import fmin_l_bfgs_b
from imageio import imwrite
import numpy as np
import time


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

def transferencia_estilo(request):
    if request.method == "POST":
        tf.compat.v1.disable_eager_execution()
        
        def Jcontent(features1, features2):
            return K.sum(K.square(features2 - features1))
        
        # Cargar los archivos
        fileObj=request.FILES['filePath']
        fs=FileSystemStorage()
        archivo=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(archivo)
        testimage='.'+filePathName

        # Hiperparametros
        imgContent_path = testimage
        imgStyle_path = "mi_negocio_web/static/img/estilo_van.jpg"
        img_height = 300
        width, height = load_img(imgContent_path).size
        print('El tamaño es ')
        print(width, height)
        img_width = int(width*img_height/height)
        alpha = 1e-2
        beta = 1e-4
        gamma = 1e-4
        content_layer = 'block5_conv2'
        style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
        iterations = 2

        def preprocess_image(image_path):
            img = load_img(image_path, target_size=(img_height, img_width))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = vgg16.preprocess_input(img)
            return img

        # Metrica autosimilitud para los mapas de rasgos
        def gram_matrix(x):
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) #aplana respecto el ancho y la altura, no el canal
            gram = K.dot(features, K.transpose(features))
            return gram

        # Mide la diferencia entre auto-similitudes de los mapas de rasgos
        def Jstyle(features1, features2):
            G1 = gram_matrix(features1)
            G2 = gram_matrix(features2)
            channels = 3
            factor =  (4.*(channels**2)*((img_height*img_width)**2))
            return K.sum(K.square(G2 - G1))/factor
        
        # Metrica para medir la similitud entre pixels contiguos en horinzotal y vertical
        def Jtotalvariation(x):
            dh = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1, :])
            dw = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, :img_height-1, 1:, :])
            return K.sum(K.pow(dh + dw, 1.25))
        
        def deprocess_image(x):
            x[:, :, 0] += 103.939  #Consultar https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            x = x[:, :, ::-1]  #Revierte BGR -> RGB
            x = np.clip(x, 0, 255).astype('uint8')
            return x

        # Cargar modelo neuronal e imágenes para su inferencia (Creando un grafo en Tensroflow)
        imgContent = K.constant(preprocess_image(imgContent_path))
        imgStyle = K.constant(preprocess_image(imgStyle_path))
        imgGen = K.placeholder((1, img_height, img_width, 3))
        input_tensor = K.concatenate([imgContent,imgStyle,imgGen], axis=0)
        model = vgg16.VGG16(input_tensor=input_tensor,weights='imagenet',include_top=False)
        #model.summary()

        loss = K.variable(0.)

        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        layer_features = outputs_dict[content_layer]
        imgContent_features = layer_features[0, :, :, :]
        imgGen_features = layer_features[2, :, :, :]
        loss = loss + alpha*Jcontent(imgContent_features,imgGen_features)

        for layer_name in style_layers:
            layer_features = outputs_dict[layer_name]
            imgStyle_features = layer_features[1, :, :, :]
            imgGen_features = layer_features[2, :, :, :]
            Jstyle_layer = Jstyle(imgStyle_features, imgGen_features)
            loss = loss + (beta/len(style_layers))*Jstyle_layer

        loss = loss + gamma*Jtotalvariation(imgGen)
    
        grads = K.gradients(loss, imgGen)[0]
        fetch_loss_and_grads = K.function([imgGen], [loss, grads])

        class Evaluator(object):
        
            def __init__(self):
                self.loss_value = None
                self.grads_values = None
                
            def loss(self, x):
                assert self.loss_value is None
                x = x.reshape((1, img_height, img_width, 3))
                outs = fetch_loss_and_grads([x])
                
                loss_value = outs[0]
                grad_values = outs[1].flatten().astype('float64')
                self.loss_value = loss_value
                self.grad_values = grad_values
                return self.loss_value

            def grads(self, x):
                assert self.loss_value is not None
                grad_values = np.copy(self.grad_values)
                self.loss_value = None
                self.grad_values = None
                return grad_values
        
        evaluator = Evaluator()

        x = preprocess_image(imgContent_path)
        x = x.flatten()

        for i in range(iterations):
            print('Iniciando iteracion', i+1)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=100)
            img = x.copy().reshape((img_height, img_width, 3))
            img = deprocess_image(img)
            end_time = time.time()
            tiempo = end_time - start_time
            
            resultado = settings.MEDIA_ROOT + '/output_' + str(i) + '.jpg'
            imwrite(resultado, img)
            #archivo = fs.open('output_' + str(i) + '.jpg')
            ruta_imagen='/media/'+ '/output_' + str(i) + '.jpg'
            
            print('Costo =', min_val)
            print('Iteracion %d completada en %dseg' % (i+1, tiempo))
        contexto = {'filePathName':filePathName,
                    'resultado':ruta_imagen,
                    'tiempo':int(tiempo)}
        return render(request, 'servicios/testilo.html', contexto)
    return render(request, 'servicios/testilo.html')