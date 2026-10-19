from django.urls import path
from . import views

urlpatterns = [
    path('',            views.home,            name='home'),
    path('about/',      views.about,           name='about'),
    path('api/data/',   views.api_filter,      name='api_filter'),
    path('api/patch/',  views.api_patch,       name='api_patch'),
    path('api/world-outline/', views.api_world_outline, name='api_world_outline'),
]