from django.urls import path
from . import views

urlpatterns = [
    path('',            views.home,            name='home'),
    path('about/',      views.about,           name='about'),
    path('classify/',   views.classify_upload, name='classify_upload'),
    path('api/data/',   views.api_filter,      name='api_filter'),
    path('api/patch/',  views.api_patch,       name='api_patch'),
]