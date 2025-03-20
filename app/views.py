import json
import os
from django.conf import settings
from django.http import JsonResponse
from django.db.models import Q
import datetime
from .models import Article, JSONData, EvaluationArticle
from .utils import reset_sequence 

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List
import asyncio

# Add NLTK just for stopwords
import nltk
from nltk.corpus import stopwords

# Import your prompts
from prompts import zero_template, one_template, few_template

def store_json_from_file(request):
    # Get the path to the JSON file in the project root
    file_path = os.path.join(settings.BASE_DIR, 'tableConvert.com_2yj0vs.json')
   
    try:
        # Delete all existing Article records and reset the sequence
        Article.objects.all().delete()
        reset_sequence(Article)

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
                    date_access = datetime.datetime.strptime(article_data['Date Access'], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Keep as None if date format is invalid
                    pass
            
            # Create a new Article object
            Article.objects.create(
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

def create_evaluation_articles(request):

    try:
        # Delete all existing EvaluationArticle records and reset the sequence
        Article.objects.all().delete()
        reset_sequence(EvaluationArticle)
        
        # Get the path to the correct JSON file in the project root
        file_path = os.path.join(settings.BASE_DIR, 'Evaluation Dataset.json')
        
        # Read JSON file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # Count of evaluation articles created
        count = 0
        
        # Create individual EvaluationArticle objects for each entry
        for article_data in data:
            # Handle date format conversion if the date is provided
            date_access = None
            if article_data.get('Date Access'):
                try:
                    date_access = datetime.datetime.strptime(article_data['Date Access'], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Keep as None if date format is invalid
                    pass
            
            # Create a new EvaluationArticle object
            EvaluationArticle.objects.create(
                # Original Article fields
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
                country_organisation=article_data.get('Country/ Organisation', ''),
                
                # New field for EvaluationArticle
                abstract=article_data.get('Abstract', ''),
                
                # Set embedding status to pending
                embedding_status='pending'
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


# Define output structure
class ResearchQuery(BaseModel):
    initial_query: str = Field(..., description="Initial search query generated from the topic")
    refined_query: str = Field(..., description="Refined search query with improved terms")
    key_concepts: List[str] = Field(..., description="Important concepts identified in the topic")
    refinement_reason: str = Field(..., description="Reasoning behind query refinement")

class ResearchOutput(BaseModel):
    result: ResearchQuery

def remove_stopwords(text):
    """Simple function to remove stopwords from text."""
    # Convert to lowercase
    text = text.lower()
    
    # Tokenise (simple space-based tokenisation)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    return filtered_words

def calculate_relevance_score(article, search_terms):

    score = 0
    
    # Process the article title using the provided remove_stopwords function.
    title_tokens = remove_stopwords(article.title) if article.title else []
    
    # Check for matches in the title (highest weight)
    for term in search_terms:
        term_lower = term.lower()
        # If the search term appears anywhere in the title string, add a higher weight.
        if article.title and term_lower in article.title.lower():
            score += 5
        # Check if the search term exactly matches any token in the processed title.
        for token in title_tokens:
            if term_lower == token:
                score += 3
    
    # Check for matches in the theme (medium weight)
    if article.theme:
        for term in search_terms:
            if term.lower() in article.theme.lower():
                score += 3
    
    # Check for matches in the research paper type (lower weight)
    if article.research_paper_type:
        for term in search_terms:
            if term.lower() in article.research_paper_type.lower():
                score += 2
    
    # Check for matches in the source (lower weight)
    if article.source:
        for term in search_terms:
            if term.lower() in article.source.lower():
                score += 1
    
    # Check for matches in the country/organisation (lower weight)
    if article.country_organisation:
        for term in search_terms:
            if term.lower() in article.country_organisation.lower():
                score += 1
    
    return score


async def refine_query_with_ollama(topic):

    prompt = ChatPromptTemplate.from_template(zero_template)
    
    # Small Model
    model = ChatOllama(**{'model': 'deepseek-r1:1.5b', 'temperature': 0.2, 'seed': 42})
    structured_llm = model.with_structured_output(ResearchOutput, method="json_schema")
    
    chain = prompt | structured_llm
    
    task = asyncio.create_task(chain.ainvoke({"topic": topic}))
    
    try:
        await asyncio.wait_for(task, 60.0)
    except asyncio.TimeoutError:
        return None
    
    return task.result().model_dump()

def process_refined_query(refined_query):

    # Remove stopwords from the whole query
    filtered_words = remove_stopwords(refined_query)
    
    # Initialise an empty Q object
    search_query = Q()
    
    if filtered_words:
        for word in filtered_words:
            if len(word) > 2:  # Only consider words with more than 2 characters
                search_query |= (
                    Q(title__icontains=word) |
                    Q(theme__icontains=word) |
                    Q(source__icontains=word) |
                    Q(research_paper_type__icontains=word) |
                    Q(country_organisation__icontains=word)
                )
        return search_query
    else:
        # If all words were stopwords, use the original refined query
        return (
            Q(title__icontains=refined_query) |
            Q(theme__icontains=refined_query) |
            Q(source__icontains=refined_query) |
            Q(research_paper_type__icontains=refined_query) |
            Q(country_organisation__icontains=refined_query)
        )


async def search_articles_async(request):

    # here for testing, when frontend is complete, this will be removed and query will be taken from frontend
    user_query = request.GET.get('query', 'prostate cancer screening guidelines')
    
    # Refine the query using Ollama
    refinement_result = await refine_query_with_ollama(user_query)
    
    if not refinement_result:
        return JsonResponse({
            'error': 'Query refinement timed out'
        }, status=504)
    
    # Get the refined query and key concepts
    refined_query = refinement_result['result']['refined_query']
    key_concepts = refinement_result['result']['key_concepts']
    
    # Process the refined query into a database query
    query_object = process_refined_query(refined_query)
    
    # Get filtered words from key concepts for debugging/display
    filtered_concepts = []
    for concept in key_concepts:
        filtered_words = remove_stopwords(concept)
        filtered_concepts.append(filtered_words)
    
    from asgiref.sync import sync_to_async

    results = await sync_to_async(list)(Article.objects.filter(query_object))
    
    # Compute search terms from the refined query (stopwords removed)
    search_terms = remove_stopwords(refined_query)
    
    # Calculate relevance score for each article and sort by score (highest first)
    scored_articles = []
    for article in results:
        score = calculate_relevance_score(article, search_terms)
        scored_articles.append((score, article))
    
    # Sort the articles by descending relevance score
    scored_articles = sorted(scored_articles, key=lambda x: x[0], reverse=True)
    
    # Limit to 20 results for performance
    articles = []
    for score, article in scored_articles[:20]:
        articles.append({
            'id': article.id,
            'title': article.title,
            'source': article.source,
            'type': article.type,
            'theme': article.theme,
            'research_paper_type': article.research_paper_type,
            'country_organisation': article.country_organisation,
            'url': article.url,
            'relevance_score': score
        })
    
    return JsonResponse({
        'query': user_query,
        'refined_query': refined_query,
        'key_concepts': key_concepts,
        'filtered_concepts': filtered_concepts,
        'result_count': len(articles),
        'articles': articles
    })

def search_articles(request):
    """Synchronous wrapper for the async search function."""
    return asyncio.run(search_articles_async(request))
