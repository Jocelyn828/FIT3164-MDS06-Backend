from django.urls import path
from . import views

urlpatterns = [
    path('store-json/', views.store_json_from_file, name='store_json'),
    path('create-evaluation-articles/', views.create_evaluation_articles, name='create_evaluation_articles'),
    path('generate-embeddings/', views.trigger_embeddings_generation, name='generate_embeddings'),
    path('api/vector-search/', views.search_articles_vector, name='api_vector_search'),
]