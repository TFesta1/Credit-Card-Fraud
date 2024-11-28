from django.urls import path
from .views import index
# from api.views import main

urlpatterns = [
    path('', index),
    path('mlFeatures', index),
    path('cohorts', index),
    path("mlAnalysis", index),
    path('room/<str:roomCode>', index)
    # path('', main)
]