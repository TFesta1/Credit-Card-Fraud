from django.shortcuts import render
from rest_framework import generics, status #Status codes
from .serializers import RoomSerializer, CreateRoomSerializer, ViewCustomerData
from .models import Room, Customer
from rest_framework.views import APIView #Parameter
from rest_framework.response import Response #Custom response
from .mlBackend.features import features


# Create your views here.
# Allows us to view a list (all Rooms)
class RoomView(generics.ListAPIView):
    queryset = Room.objects.all() #All room objects
    serializer_class = RoomSerializer #Serializer class (To JSON)
    
class GetRoom(APIView):
    serializer_class = RoomSerializer
    lookup_url_kwarg = 'code' #Pass code

    def get(self, request, format=None):
        code = request.GET.get(self.lookup_url_kwarg) #Get the code from the URL
        if code != None: #If the code is not None
            room = Room.objects.filter(code=code)
            if len(room) > 0:
                data = RoomSerializer(room[0]).data 
                data['is_host'] = self.request.session.session_key == room[0].host #Check if the current user is the host.
                return Response(data, status=status.HTTP_200_OK)
            return Response({'Room Not Found': 'Invalid Room Code.'}, status=status.HTTP_404_NOT_FOUND)
        return Response({'Bad Request': 'Code parameter not found in request'}, status=status.HTTP_400_BAD_REQUEST)

class JoinRoom(APIView):
    lookup_url_kwarg = 'code'

    def post(self, request, format=None):
        if not self.request.session.exists(self.request.session.session_key): #Checks if current user has active Session (remembering login)
            self.request.session.create()

        code = request.data.get(self.lookup_url_kwarg) #Post requests use the .data field
        if code != None:
            room_result = Room.objects.filter(code=code)
            if len(room_result) > 0:
                room = room_result[0]
                self.request.session['room_code'] = code #This user is IN this room. If they are, and come back to the website, it returns them to the room
                return Response({'message': 'Room Joined!'}, status=status.HTTP_200_OK)
            return Response({'Bad Request': 'Invalid Room Code'}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'Bad Request': 'Invalid post data, did not find a code key'}, status=status.HTTP_400_BAD_REQUEST)

class CreateRoomView(APIView):
    serializer_class = CreateRoomSerializer

    def post(self, request, format=None):
        if not self.request.session.exists(self.request.session.session_key): #Checks if current user has active Session (remembering login)
            self.request.session.create()

        serializer = self.serializer_class(data=request.data) #Checks if data sent is valid
        if serializer.is_valid(): #If the data is in the post request
            guest_can_pause = serializer.data.get('guest_can_pause')
            votes_to_skip = serializer.data.get('votes_to_skip')
            host = self.request.session.session_key
            queryset = Room.objects.filter(host=host) #Any rooms in db that have the same host 
            if queryset.exists(): #Do not create a new room if it exists already
                room = queryset[0] #Grab active room that exists
                room.guest_can_pause = guest_can_pause
                room.votes_to_skip = votes_to_skip
                self.request.session['room_code'] = room.code
                room.save(update_fields=['guest_can_pause', 'votes_to_skip'])
            else:
                room = Room(host=host, guest_can_pause=guest_can_pause, votes_to_skip=votes_to_skip)
                room.save()
                self.request.session['room_code'] = room.code
            return Response(RoomSerializer(room).data, status=status.HTTP_201_CREATED)

class GetCustomer(APIView):
    serializer_class = ViewCustomerData
    def post(self, request, format=None):
        if not self.request.session.exists(self.request.session.session_key): #Checks if current user has active Session (remembering login)
            self.request.session.create()
            serializer = self.serializer_class(data=request.data) #Checks if data sent is valid, being id...
            if serializer.is_valid():
                customer_id = serializer.data.get('customer_id')
                customer = Customer(customer_id=customer_id)
                customer.save()
                return Response(ViewCustomerData(customer).data, status=status.HTTP_200_OK)
            return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)


class GetFeatures(APIView):
    def get(self, request, format=None):
        featureList = features()
        return Response(featureList, status=status.HTTP_200_OK)