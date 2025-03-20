from django.urls import path
from . import views

urlpatterns = [
    path('store-json/', views.store_json_from_file, name='store_json'),
    path('api/search/', views.search_articles, name='api_search'),
    # New URL for creating evaluation articles
    path('create-evaluation-articles/', views.create_evaluation_articles, name='create_evaluation_articles'),
]