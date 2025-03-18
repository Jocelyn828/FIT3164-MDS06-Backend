import json
import os
from django.conf import settings
from django.http import JsonResponse
from .models import JSONData, Article
from datetime import datetime

def store_json_from_file(request):
    # Get the path to the JSON file in the project root
    file_path = os.path.join(settings.BASE_DIR, 'tableConvert.com_2yj0vs.json')
   
    try:
        # Read JSON file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
       
        # Count of articles created
        count = 0
        
        # Create individual Article objects for each entry
        for article_data in data:
            # Handle date format conversion if the date is provided
            date_access = None
            if article_data.get('Date Access'):
                try:
                    date_access = datetime.strptime(article_data['Date Access'], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Keep as None if date format is invalid
                    pass
            
            # Create a new Article object
            article = Article.objects.create(
                source=article_data.get('Source', ''),
                type=article_data.get('Type', ''),
                date_access=date_access,
                url=article_data.get('URL', ''),
                final_level_1_consensus=article_data.get('Final Level 1 Consensus', ''),
                exclusion_reason_final_level_1=article_data.get('Exclusion Reason Final Level 1', ''),
                final_level_2_consensus=article_data.get('Final Level 2 Consensus', ''),
                exclusion_reason_final_level_2=article_data.get('Exclusion Reason Final Level 2', ''),
                title=article_data.get('Title', ''),
                theme=article_data.get('Theme', ''),
                research_paper_type=article_data.get('Research Paper Type', ''),
                country_organisation=article_data.get('Country/ Organisation', '')
            )
            count += 1
       
        return JsonResponse({
            'message': f'JSON file processed successfully. Created {count} articles.',
            'articles_created': count
        })
    except FileNotFoundError:
        return JsonResponse({'error': f'JSON file not found: {file_path}'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Error processing file: {str(e)}'}, status=500)