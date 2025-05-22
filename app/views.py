import json
import os
import datetime
import asyncio

import numpy as np
import re

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
from .utils import reset_sequence, extract_text_from_document

from prompts import zero_template, boolean_refine_query_template, keyword_expansion_template, keyword_generation_template


class ExclusionAnalysis(BaseModel):
    classification: str
    keywords: List[str]
    reason: str


class ExclusionAnalysisOutput(BaseModel):
    result: ExclusionAnalysis


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


class KeywordResults(BaseModel):
    keywords: List[str] = Field(..., description="List of relevant keywords extracted from the query")

class KeywordOutput(BaseModel):
    result: KeywordResults

def validate_query(query):
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    # Count words
    word_count = len([word for word in query.split() if word.strip()])
    
    if word_count > 500:  # Maximum 50 words
        return False, "Query is too long. Maximum 500 words allowed."
    
    return True, ""

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
    prompt = ChatPromptTemplate.from_template(boolean_refine_query_template)
    
    model = ChatOllama(**{'model': 'deepseek-r1:1.5b', 'temperature': 0.2, 'seed': 42})
    structured_llm = model.with_structured_output(ResearchOutput, method="json_schema")
    
    chain = prompt | structured_llm
    
    task = asyncio.create_task(chain.ainvoke({"text": topic}))
    
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


async def generate_keywords_with_ollama(query):
    """
    Use Ollama LLM to generate keywords from a user query
    """
    prompt = ChatPromptTemplate.from_template(keyword_generation_template)
    
    model = ChatOllama(**{'model': 'deepseek-r1:1.5b', 'temperature': 0.1, 'seed': 42})
    structured_llm = model.with_structured_output(KeywordOutput, method="json_schema")
    
    chain = prompt | structured_llm
    
    task = asyncio.create_task(chain.ainvoke({"query": query}))
    
    try:
        # Set timeout for the task
        await asyncio.wait_for(task, 30.0)
    except asyncio.TimeoutError:
        return None
    
    return task.result().model_dump()


async def analyze_document_with_ollama(criteria: dict, document_text: str) -> dict:
    """
    Use Ollama LLM to analyze a document against exclusion criteria
    """
    # Debug: Print document text info
    print(f"[DEBUG] Document text length: {len(document_text)} characters")
    print(f"[DEBUG] Document text preview (first 200 chars): {document_text[:200]}")
    
    # Debug: Check if document text is empty or very short
    if len(document_text) < 50:
        print(f"[ERROR] Document text is suspiciously short: {document_text}")
    
    # Debug: Print criteria
    print(f"[DEBUG] Criteria received: {criteria}")
    
    # Trim document text if too long
    max_text_length = 4000  
    trimmed_text = document_text[:max_text_length] + ("..." if len(document_text) > max_text_length else "")
    
    print(f"[DEBUG] Trimmed text length: {len(trimmed_text)} characters")
    print(f"[DEBUG] Trimmed text preview (first 100 chars): {trimmed_text[:100]}")
    print(f"[DEBUG] Trimmed text preview (last 100 chars): {trimmed_text[-100:] if len(trimmed_text) > 100 else trimmed_text}")
    
    prompt_text = """
    ## Prompt Description
    You are reviewing a medical document about prostate cancer that has been classified for **EXCLUSION**.

    ---

    ## Document Text
    ```text
    {text}
    ````

    ---

    ## Input Data
    ## Exclusion Reason

    ```json
    {criteria_json}
    ```

    ---

    ## Task Instructions
    1. Interpret what the **exclusion reason** likely means in context
    2. Read through the document carefully and explain **why** it meets the reason for exclusion
    3. IMPORTANT: Extract AT LEAST 3-5 specific words or phrases DIRECTLY FROM THE DOCUMENT TEXT that justify the exclusion. If it's a non-English document, extract some foreign words or phrases from the text as evidence.

    ---

    ## Response Format
    ```json
    {{
    "classification": "The given exclusion reason",
    "keywords": ["exact", "words", "or", "phrases", "from", "the", "document", "that", "justify", "exclusion"],
    "reason": "A detailed explanation of why this document should be excluded based on the given reason"
    }}
    ```

    ---

    ## Example 1
    **Exclusion reason:**
    ```
    "Non-English"
    ```

    **Document text:**
    ```
    "Ce document traite des approches nutritionnelles dans le traitement du cancer de la prostate avancé chez les patients âgés."
    ```

    **Expected output:**
    ```json
    {{
    "classification": "Non-English",
    "keywords": ["Ce document", "approches nutritionnelles", "traitement", "cancer de la prostate", "patients âgés"],
    "reason": "The document is written in French, which does not meet the English language requirement for inclusion."
    }}
    ```

    ---

    ## Final Note
    Return **only** the JSON with **no additional text**.
    """

    # Create prompt template with the appropriate variables
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    # Convert criteria dict to JSON string
    criteria_json = json.dumps(criteria, indent=2)
    
    # Debug: Print prompt variables
    print(f"[DEBUG] criteria_json: {criteria_json[:100]}...")
    
    model = ChatOllama(**{'model': 'gemma:7b', 'temperature': 0.2, 'seed': 42})
    
    # Debug: Print model info
    print(f"[DEBUG] Using model: {model.model}")
    
    # Set up structured output using the ExclusionAnalysisOutput Pydantic model
    structured_llm = model.with_structured_output(ExclusionAnalysisOutput, method="json_schema")
    
    # Create a chain
    chain = prompt | structured_llm
    
    # Prepare the inputs
    inputs = {
        "text": trimmed_text, 
        "criteria_json": criteria_json
    }
    
    # Debug: Print final inputs
    print(f"[DEBUG] Final input: text length={len(inputs['text'])}, criteria_json length={len(inputs['criteria_json'])}")
    
    # Invoke the chain asynchronously
    print("[DEBUG] Starting model invocation...")
    task = asyncio.create_task(chain.ainvoke(inputs))
    
    try:
        # Set a timeout for the task
        print("[DEBUG] Waiting for model response...")
        await asyncio.wait_for(task, 60.0)
        print("[DEBUG] Model response received!")
    except asyncio.TimeoutError:
        print("[ERROR] Model invocation timed out after 60 seconds")
        return None
    except Exception as e:
        print(f"[ERROR] Exception in model invocation: {str(e)}")
        return None
    
    # Get the result
    result = task.result().model_dump()
    
    # Debug: Print the raw result
    print(f"[DEBUG] Raw result from model: {result}")
    
    # Debug: Check keywords
    if 'result' in result and 'keywords' in result['result']:
        keywords = result['result']['keywords']
        print(f"[DEBUG] Keywords extracted: {keywords}")
        
        # Check if keywords are actually from the document
        for keyword in keywords:
            if keyword.lower() in document_text.lower():
                print(f"[DEBUG] Keyword '{keyword}' FOUND in document")
            else:
                print(f"[DEBUG] Keyword '{keyword}' NOT FOUND in document")
    
    return result


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
    
    # Check if 'title' appears in the query
    has_title_in_query = 'title' in user_query.lower()
    
    # If 'title' is in the query, extract the titles to exclude
    exclude_articles = []
    if has_title_in_query:
        exclude_articles = re.findall(r'title:\s*"([^"]+)"', user_query)

    # Calculate similarity scores for each article
    scored_articles = []
    for article in articles:

        # Exclude if the article is in the exclude list            
        if has_title_in_query and article.title:
            if article.title in exclude_articles:
                continue
            
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
def generate_keywords_view(request):
    """Synchronous wrapper for the async keyword generation function."""
    return asyncio.run(generate_keywords(request))


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


@csrf_exempt
@require_http_methods(["GET", "POST"])
async def generate_keywords(request):
    """
    Generate key concepts/keywords based on the user's initial query
    """
    if request.method == 'GET':
        user_query = request.GET.get('query', '')
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_query = data.get('query', '')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    # Validate query
    is_valid, error_message = validate_query(user_query)
    if not is_valid:
        return JsonResponse({
            'error': error_message
        }, status=400)
    
    # Check cache first for faster responses
    cache_key = f'keywords:{user_query.lower().strip()}'
    cached_result = cache.get(cache_key)
    
    if cached_result:
        return JsonResponse({
            'success': True,
            'query': user_query,
            'keywords': cached_result
        })
    
    # If not in cache, use the LLM to generate keywords
    try:
        # FIXED: Use the correct function
        keyword_result = await generate_keywords_with_ollama(user_query)
        
        if not keyword_result:
            return JsonResponse({
                'error': 'Keyword generation timed out'
            }, status=504)
        
        # FIXED: Extract keywords from the correct result structure
        keywords = keyword_result['result']['keywords']
        
        # Cache the result for future use (1 day)
        cache.set(cache_key, keywords, 86400)
        
        return JsonResponse({
            'success': True,
            'query': user_query,
            'keywords': keywords
        })
    except Exception as e:
        return JsonResponse({
            'error': f'Error generating keywords: {str(e)}'
        }, status=500)
    

@csrf_exempt
@require_http_methods(["POST"])
def analyze_exclusion(request):
    try:
        # Check if document was uploaded
        if 'document' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No document uploaded'
            }, status=400)
        
        document = request.FILES['document']
        
        # Get exclusion criteria from form data or JSON
        exclusion_criteria = request.POST.get('criteria', '')
        if not exclusion_criteria and request.content_type == 'application/json':
            try:
                body_data = json.loads(request.body)
                exclusion_criteria = body_data.get('criteria', '')
            except json.JSONDecodeError:
                pass
        
        if not exclusion_criteria:
            return JsonResponse({
                'success': False,
                'error': 'No exclusion criteria provided'
            }, status=400)
        
        # Extract text from the document
        document_text = extract_text_from_document(document)
        if not document_text:
            return JsonResponse({
                'success': False,
                'error': 'Could not extract text from document'
            }, status=400)
        
        # *** DEBUG PRINTS ***
        print(f"[analyze_exclusion] Exclusion criteria: {exclusion_criteria!r}")
        print(f"[analyze_exclusion] Document text preview: {document_text[:500]!r}")
        
        # Generate a cache key based on criteria and document content hash
        cache_key = f'exclusion_analysis:{hash(exclusion_criteria + document_text[:1000])}'
        cached_result = cache.get(cache_key)
        if cached_result:
            return JsonResponse({
                'success': True,
                'documentName': document.name,
                'result': cached_result
            })
        
        # Analyze the document
        analysis_result = asyncio.run(analyze_document_with_ollama(exclusion_criteria, document_text))
        if not analysis_result:
            return JsonResponse({
                'success': False,
                'error': 'Document analysis timed out'
            }, status=504)
        
        # Extract the result
        result = analysis_result['result']
        
        # Cache the result for future use (1 day)
        cache.set(cache_key, result, 86400)
        
        return JsonResponse({
            'success': True,
            'documentName': document.name,
            'result': result
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error analyzing document exclusion: {str(e)}'
        }, status=500)
