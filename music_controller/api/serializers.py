# This takes our model (Room), has code, translates into JSON Response
from rest_framework import serializers
from .models import Room, Customer


class RoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room
        fields = ('id', 'code', 'host', 'guest_can_pause', 'votes_to_skip', 'created_at')

class CreateRoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room 
        fields = ('guest_can_pause', 'votes_to_skip')

# Fields sent as part of the post request. Probably will need more fields.
class ViewCustomerData(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = ('customer_id',)