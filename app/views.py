import json
import os
import datetime
import asyncio

import numpy as np

from django.conf import settings
from django.http import JsonResponse
from django.db.models import Q
from asgiref.sync import sync_to_async
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.cache import cache

from pydantic import BaseModel, Field
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .models import Article, JSONData, EvaluationArticle
from .utils import reset_sequence

from prompts import zero_template, one_template, few_template, keyword_expansion_template



class ResearchQuery(BaseModel):
    initial_query: str = Field(..., description="Initial search query generated from the topic")
    refined_query: str = Field(..., description="Refined search query with improved terms")
    key_concepts: List[str] = Field(..., description="Important concepts identified in the topic")
    refinement_reason: str = Field(..., description="Reasoning behind query refinement")


class ResearchOutput(BaseModel):
    result: ResearchQuery


class ExpandedQuery(BaseModel):
    expanded_query: str = Field(..., description="Expanded search query created from the keyword")


class QueryExpansionResult(BaseModel):
    result: ExpandedQuery


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
        EvaluationArticle.objects.all().delete()  
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


def generate_embedding(text, model_name="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings.embed_query(text)


def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    # Handle zero division
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)


async def generate_embeddings_for_all_articles(request):
    try:
        # Get all articles with pending embedding status
        articles = await sync_to_async(list)(EvaluationArticle.objects.filter(embedding_status='pending'))
        
        processed_count = 0
        failed_count = 0
        
        for article in articles:
            try:
                # Only process if article has an abstract
                if article.abstract:
                    # Generate embedding for the abstract
                    embedding = generate_embedding(article.abstract)
                    
                    # Update the article with the embedding
                    article.embedding = embedding
                    article.embedding_status = 'completed'
                else:
                    article.embedding_status = 'failed'
                    failed_count += 1
                
                # Save the article
                await sync_to_async(article.save)()
                processed_count += 1
                
            except Exception as e:
                # Update status to failed if there's an error
                article.embedding_status = 'failed'
                await sync_to_async(article.save)()
                failed_count += 1
        
        return JsonResponse({
            'message': f'Processed {processed_count} articles. Failed: {failed_count}',
            'processed': processed_count,
            'failed': failed_count
        })
    
    except Exception as e:
        return JsonResponse({'error': f'Error generating embeddings: {str(e)}'}, status=500)


async def refine_query_with_ollama(topic):
    prompt = ChatPromptTemplate.from_template(zero_template)
    
    model = ChatOllama(**{'model': 'deepseek-r1:1.5b', 'temperature': 0.2, 'seed': 42})
    structured_llm = model.with_structured_output(ResearchOutput, method="json_schema")
    
    chain = prompt | structured_llm
    
    task = asyncio.create_task(chain.ainvoke({"topic": topic}))
    
    try:
        await asyncio.wait_for(task, 60.0)
    except asyncio.TimeoutError:
        return None
    
    return task.result().model_dump()


async def expand_keyword_with_ollama(keyword):
    """
    Use Ollama LLM to expand a keyword into a full search query
    """
    prompt = ChatPromptTemplate.from_template(keyword_expansion_template)
    
    model = ChatOllama(**{'model': 'deepseek-r1:1.5b', 'temperature': 0.2, 'seed': 42})
    structured_llm = model.with_structured_output(QueryExpansionResult, method="json_schema")
    
    chain = prompt | structured_llm
    
    task = asyncio.create_task(chain.ainvoke({"keyword": keyword}))
    
    try:
        # Shorter timeout for this simpler task
        await asyncio.wait_for(task, 30.0)
    except asyncio.TimeoutError:
        return None
    
    return task.result().model_dump()


async def search_articles_vector_async(request):
    if request.method == 'GET':
        user_query = request.GET.get('query', 'prostate cancer screening guidelines')
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_query = data.get('message', '')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    refinement_result = await refine_query_with_ollama(user_query)
    
    if not refinement_result:
        return JsonResponse({
            'error': 'Query refinement timed out'
        }, status=504)
    
    # Get the refined query and key concepts
    refined_query = refinement_result['result']['refined_query']
    key_concepts = refinement_result['result']['key_concepts']
    
    # Generate embedding for the refined query
    query_embedding = generate_embedding(refined_query)
    
    # Get all articles with completed embeddings
    articles = await sync_to_async(list)(
        EvaluationArticle.objects.filter(embedding_status='completed')
    )
    
    # Calculate similarity scores for each article
    scored_articles = []
    for article in articles:
        if article.embedding:
            # Calculate cosine similarity between query and article embeddings
            similarity_score = cosine_similarity(query_embedding, article.embedding)
            scored_articles.append((similarity_score, article))
    
    # Sort articles by similarity score (highest first)
    scored_articles.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 20 results
    results = []
    for score, article in scored_articles[:20]:
        results.append({
            'id': article.id,
            'title': article.title,
            'source': article.source,
            'type': article.type,
            'theme': article.theme,
            'research_paper_type': article.research_paper_type,
            'country_organisation': article.country_organisation,
            'url': article.url,
            'similarity_score': float(score),
            'abstract_preview': article.abstract[:200] + '...' if len(article.abstract) > 200 else article.abstract
        })
    
    return JsonResponse({
        'query': user_query,
        'refined_query': refined_query,
        'key_concepts': key_concepts,
        'result_count': len(results),
        'articles': results
    })

@csrf_exempt
def search_articles_vector(request):
    """Synchronous wrapper for the async vector search function."""
    return asyncio.run(search_articles_vector_async(request))


def trigger_embeddings_generation(request):
    """Endpoint to trigger the embedding generation process."""
    return asyncio.run(generate_embeddings_for_all_articles(request))


@csrf_exempt
@require_http_methods(["POST"])
def expand_keyword(request):
    try:
        # Parse the request body
        data = json.loads(request.body)
        keyword = data.get('keyword', '')
        
        if not keyword:
            return JsonResponse({
                'success': False,
                'error': 'No keyword provided'
            }, status=400)
        
        # Check cache first
        cache_key = f'expanded_keyword:{keyword.lower().strip()}'
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return JsonResponse({
                'success': True,
                'originalKeyword': keyword,
                'expandedQuery': cached_result
            })
        
        # If not in cache, use the LLM to expand
        expansion_result = asyncio.run(expand_keyword_with_ollama(keyword))
        
        if not expansion_result:
            return JsonResponse({
                'success': False,
                'error': 'Query expansion timed out'
            }, status=504)
        
        # Extract the expanded query from the result
        expanded_query = expansion_result['result']['expanded_query']
        
        # Cache the result for future use (1 day)
        cache.set(cache_key, expanded_query, 86400)
        
        # Return the expanded query
        return JsonResponse({
            'success': True,
            'originalKeyword': keyword,
            'expandedQuery': expanded_query
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error expanding keyword: {str(e)}'
        }, status=500)