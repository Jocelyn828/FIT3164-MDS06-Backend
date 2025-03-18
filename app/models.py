from django.db import models

class Article(models.Model):
    source = models.CharField(max_length=255, blank=True)
    type = models.CharField(max_length=255, blank=True)
    date_access = models.DateTimeField(null=True, blank=True)
    url = models.TextField(blank=True)  
    final_level_1_consensus = models.CharField(max_length=255, blank=True)
    exclusion_reason_final_level_1 = models.TextField(blank=True)  
    final_level_2_consensus = models.CharField(max_length=255, blank=True)
    exclusion_reason_final_level_2 = models.TextField(blank=True) 
    title = models.TextField(blank=True)  
    theme = models.CharField(max_length=255, blank=True)
    research_paper_type = models.CharField(max_length=255, blank=True)
    country_organisation = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title[:50] if self.title else f"Article {self.id}"

# In case I need it in the future (not used for now)
class JSONData(models.Model):
    data = models.JSONField()
    name = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
   
    def __str__(self):
        return self.name or f"JSONData {self.id}"