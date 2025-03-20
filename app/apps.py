from django.apps import AppConfig
import nltk

class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self):
        # Download necessary NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
