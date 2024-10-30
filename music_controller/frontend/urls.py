from django.urls import path
from .views import index
# from api.views import main

urlpatterns = [
    path('', index)
    # path('', main)
]