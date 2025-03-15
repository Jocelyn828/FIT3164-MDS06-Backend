import json
import os
from django.conf import settings
from django.http import JsonResponse
from .models import JSONData

def store_json_from_file(request):
    # Get the path to the JSON file in the project root
    file_path = os.path.join(settings.BASE_DIR, 'tableConvert.com_2yj0vs.json')
   
    try:
        # Read JSON file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
       
        # Save JSON content to the database
        json_obj = JSONData.objects.create(data=data)
       
        return JsonResponse({
            'message': 'JSON file stored successfully',
            'id': json_obj.id,
            'created_at': json_obj.created_at.isoformat()
        })
    except FileNotFoundError:
        return JsonResponse({'error': f'JSON file not found: {file_path}'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
