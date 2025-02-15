from django.apps import AppConfig
from .mlBackend.mlModel import trainModel
import os

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    def ready(self):
        if os.environ.get('RUN_MAIN'):
            trainModel()
