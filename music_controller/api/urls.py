from django.urls import path
from .views import RoomView


urlpatterns = [
    path('room', RoomView.as_view())
]


# python .\manage.py makemigrations
# python .\manage.py migrate
#  Whenever we make a change to the model, or database, run this.

# OR run it when we first boot up the app


# Running server
# python .\manage.py runserver

# 16:17
"""
Errors and solutions:
Error: No such room _
Solution: python .\manage.py makemigrations and python .\manage.py migrate

Tips:


"""