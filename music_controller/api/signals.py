# from django.apps import AppConfig, ready
# from django.dispatch import receiver
# from .mlBackend.mlModel import trainModel

# # On server start, load this data
# @receiver(ready)
# def load_data_on_start(sender, **kwargs):
#     trainModel()


# from django.apps import AppConfig
# from django.apps.signals import ready
# from django.dispatch import receiver
# from .mlBackend.features import fetch_data

# @receiver(ready)
# def load_data_on_start(sender, **kwargs):
#     fetch_data()
