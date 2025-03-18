from django.contrib import admin
from .models import JSONData, Article
from django.forms import widgets
import json
from django.db import models

# Fixed JSONField widget to display JSON data in a pretty format for better readability
class PrettyJSONWidget(widgets.Textarea):
    def format_value(self, value):
        try:
            value = json.dumps(json.loads(value), indent=2, sort_keys=True)
            # these lines will try to adjust size of TextArea to fit to content
            row_lengths = [len(r) for r in value.split('\n')]
            self.attrs['rows'] = min(max(len(row_lengths) + 2, 10), 30)
            self.attrs['cols'] = min(max(max(row_lengths) + 2, 40), 120)
            return value
        except Exception as e:
            return super(PrettyJSONWidget, self).format_value(value)

# In case I need it in the future (not used for now)  
@admin.register(JSONData)
class JSONDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at')
    readonly_fields = ('created_at',)
    formfield_overrides = {
        models.JSONField: {'widget': PrettyJSONWidget}
    }

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'source', 'type', 'theme', 'created_at')
    list_filter = ('source', 'type', 'theme', 'research_paper_type')
    search_fields = ('title', 'source', 'theme', 'country_organisation')
    readonly_fields = ('created_at',)