from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
from mailchimp_marketing import Client
from mailchimp_marketing.api_client import ApiClientError
from django.core.mail import send_mail

api_key = settings.MAILCHIMP_API_KEY
server = settings.MAILCHIMP_DATA_CENTER
list_id = settings.MAILCHIMP_EMAIL_LIST_ID

def suscripcion(request):
    if request.method == "POST":
        email = request.POST['email']

        mailchimp = Client()
        mailchimp.set_config({
            "api_key": api_key,
            "server": server,
        })

        member_info = {
            "email_address": email,
            "status": "subscribed",
        }

        try:
            response = mailchimp.lists.add_list_member(list_id, member_info)
            messages.success(request, "Email recibido. ¡Gracias! ")
            #asunto = "Suscripción"
            #mensaje = "Gracias por suscribirte a nuestro boletin"
            #remitente = "django.actumlogos@gmail.com"
            #destinatario = email
            #send_mail(asunto, mensaje, remitente, [destinatario])
        except ApiClientError as error:
            messages.success(request, "Hubo un problema. Intentelo más tarde")
    
    return render(request, "marketing/email_sus.html")