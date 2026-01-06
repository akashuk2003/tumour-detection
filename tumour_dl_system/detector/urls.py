from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from detector import views 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload/', views.upload_view, name='upload'),
    path('compare/', views.comparison_view, name='compare'), 
    path('', views.upload_view),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)