from django.urls import path
from .views import RoomView, CreateRoomView, GetRoom, JoinRoom


urlpatterns = [
    path('room', RoomView.as_view()),
    path('create-room', CreateRoomView.as_view()),
    path('get-room', GetRoom.as_view()),
    path('join-room', JoinRoom.as_view())
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

Error: Failed when doing npm i @material-ui/core 
node_modules/react
npm ERR!   dev react@"^17.0.0" from the root project
npm ERR!   peer react@"^16.8.0 || ^17.0.0" from @material-ui/core@4.12.4
npm ERR!   node_modules/@material-ui/core
npm ERR!     @material-ui/core@"*" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^18.3.1" from react-dom@18.3.1
npm ERR! node_modules/react-dom
npm ERR!   dev react-dom@"^18.3.1" from the root project
Solution: Do npm install react-dom@^17.0.0 and npm install react@^17.0.0 since it's asking for that version, then do npm i @material-ui/core 

Error: Templates does not exist
Solution: 

Tips:


"""