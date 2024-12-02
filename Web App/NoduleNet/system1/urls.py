from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('home', views.home, name='home'),
    path('about', views.about, name='about'),
    path('detect', views.detect, name='detect'),
    path('process', views.process, name='process'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)