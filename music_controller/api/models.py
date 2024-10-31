from django.db import models
import string 
import random 

def generate_unique_code():
    length = 6
    while True:
        code = ''.join(random.choices(string.ascii_uppercase, k=length)) #Random code at k length, only contains uppercase ascii chars
        if Room.objects.filter(code=code).count() == 0:
            break
    return code

# Create your models here.
# Table, rows, cols, to store info. A table = model. Write python code to do DB operations

# Group users together in a room
class Room(models.Model): #From DJango documentation
    code = models.CharField(max_length=8, default=generate_unique_code, unique=True) # 8 char long, unique, default empty string
    host = models.CharField(max_length=50, unique=True) # host of the room, at most one host (unique=True)
    guest_can_pause = models.BooleanField(null=False, default=False) # Guess must pass a value
    votes_to_skip = models.IntegerField(null=False, default=1)
    created_at = models.DateTimeField(auto_now_add=True) #We never have to pass the DT, it automatically passes the datetime it was added

    # def is_host ...
    # Fat models, thin views

class Customer(models.Model):
    customer_id = models.CharField(max_length=50, primary_key=True, unique=True)