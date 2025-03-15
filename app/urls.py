from django.urls import path
from . import views

urlpatterns = [
    path('store-json/', views.store_json_from_file, name='store_json'),
]