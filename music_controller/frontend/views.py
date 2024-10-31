from django.shortcuts import render

# Create your views here.
# Render template, then let react take care of it
def index(request, *args, **kwargs):
    return render(request, 'frontend/index.html') #Render the index.html file in the frontend folder