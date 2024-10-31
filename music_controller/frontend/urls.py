from django.urls import path
from .views import index
# from api.views import main

urlpatterns = [
    path('', index),
    path('join', index),
    path('create', index),
    path('room/<str:roomCode>', index)
    # path('', main)
]