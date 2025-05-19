import os
import django
import asyncio
from pprint import pprint

# Set up Django environment BEFORE importing any Django modules
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

# Now we can import Django modules
from app.views import (
    generate_keywords_with_ollama,
    refine_query_with_ollama,
    expand_keyword_with_ollama,
    search_articles_vector_async
)

def print_test_header(function_name):
    print("\n" + "="*50)
    print(f"TESTING: {function_name}")
    print("="*50)

def print_test_result(input_data, output_data):
    print("\nINPUT:")
    pprint(input_data, width=125)
    print("\nOUTPUT:")
    if isinstance(output_data, dict):
        if 'key_concepts' in output_data:
            for key in ['query', 'refined_query', 'key_concepts', 'result_count', 'articles']:
                if key == 'articles':
                    print("articles:", end=" ")
                    pprint(output_data['articles'], width=125)
                pprint(f"{key}: {output_data[key]}", width=125)
        else:
            pprint(output_data, width=125)
            # pprint(output_data)
    else:
        pprint(output_data, width=125)
        # pprint(output_data)
    print("\n" + "="*50)
    print("\n")

async def test_generate_keywords():
    print_test_header("GENERATE KEYWORDS")
    input_query = "The NCCN Guidelines for Prostate Cancer Early Detection provide recommendations for men choosing to participate in an early detection program for prostate cancer. These NCCN Guidelines Insights highlight notable recent updates. Overall, the 2014 update represents a more streamlined and concise set of recommendations. The panel stratified the age ranges at which initiating testing for prostate cancer should be considered. Indications for biopsy include both a cutpoint and the use of multiple risk variables in combination. In addition to other biomarkers of specificity, the Prostate Health Index has been included to aid biopsy decisions in certain men, given recent FDA approvals."
    output = await generate_keywords_with_ollama(input_query)
    print_test_result(input_query, output)

async def test_refine_query():
    print_test_header("REFINE QUERY")
    input_topic = "prostate cancer screening guidelines" 
    output = await refine_query_with_ollama(input_topic)
    print_test_result(input_topic, output)

async def test_expand_keyword():
    print_test_header("EXPAND KEYWORD")
    input_keyword = "prostate cancer"
    output = await expand_keyword_with_ollama(input_keyword)
    print_test_result(input_keyword, output)

async def test_vector_search():
    print_test_header("VECTOR SEARCH")
    input_query = "prostate cancer"
    output = await search_articles_vector_async(input_query)
    print_test_result(input_query, output)

if __name__ == "__main__":
    # run the test individually
    # asyncio.run(test_generate_keywords())
    # asyncio.run(test_refine_query())
    asyncio.run(test_expand_keyword())
    # asyncio.run(test_vector_search())
    
    # # Run all tests
    # async def run_all_tests():
    #     await test_generate_keywords()
    #     await test_refine_query()
    #     await test_expand_keyword()
    #    await test_vector_search()
    
    # asyncio.run(run_all_tests())