from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'famcam'

urlpatterns = [
    path('', views.famcam, name='main'),
    path('learnphoto/', views.learnphoto, name='learnphoto'),
    path('startcam/', views.startcam, name='startcam'),
]