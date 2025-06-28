# cyst_detection/urls.py

from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from detection import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.upload_and_detect, name='upload_and_detect'),
    path('', views.upload_and_detect, name='upload'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
