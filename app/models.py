from django.db import models

class JSONData(models.Model): 
    data = models.JSONField()
    name = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name or f"JSONData {self.id}"