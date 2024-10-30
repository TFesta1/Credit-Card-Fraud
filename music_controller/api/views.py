from django.shortcuts import render
from rest_framework import generics
from .serializers import RoomSerializer
from .models import Room

# Create your views here.
# Allows us to view a list (all Rooms)
class RoomView(generics.ListAPIView):
    queryset = Room.objects.all() #All room objects
    serializer_class = RoomSerializer #Serializer class (To JSON)
    

