import os
import django
import asyncio
from pprint import pprint
from django.core.files.uploadedfile import SimpleUploadedFile

# Set up Django environment BEFORE importing any Django modules
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

# Import Django modules
from app.views import (
    generate_keywords_with_ollama,
    refine_query_with_ollama,
    expand_keyword_with_ollama,
    search_articles_vector_async,
    validate_query,
)

from app.utils import extract_text_from_document

def print_test_header(function_name):
    print("\n" + "="*50)
    print(f"TESTING: {function_name}")
    print("="*50)

def print_test_result(input_data, output_data):
    print("\nINPUT:")
    pprint(input_data, width=125)
    print("\nOUTPUT:")
    pprint(output_data, width=125)
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

def test_extract_text_from_document():
    print_test_header("EXTRACT TEXT FROM DOCUMENT")
    text_content = "This is a test text file.\nIt has multiple lines.\nTesting text extraction."
    text_file = SimpleUploadedFile("test.txt", text_content.encode('utf-8'))
    output = extract_text_from_document(text_file)
    print_test_result("Text file", output)

def test_validate_query():
    print_test_header("VALIDATE QUERY")
    input = "The NCCN Guidelines for Prostate Cancer Early Detection provide recommendations for men choosing to participate in an early detection program for prostate cancer. These NCCN Guidelines Insights highlight notable recent updates. Overall, the 2014 update represents a more streamlined and concise set of recommendations. The panel stratified the age ranges at which initiating testing for prostate cancer should be considered. Indications for biopsy include both a cutpoint and the use of multiple risk variables in combination. In addition to other biomarkers of specificity, the Prostate Health Index has been included to aid biopsy decisions in certain men, given recent FDA approvals.The NCCN Guidelines for Prostate Cancer Early Detection provide recommendations for men choosing to participate in an early detection program for prostate cancer. These NCCN Guidelines Insights highlight notable recent updates. Overall, the 2014 update represents a more streamlined and concise set of recommendations. The panel stratified the age ranges at which initiating testing for prostate cancer should be considered. Indications for biopsy include both a cutpoint and the use of multiple risk variables in combination. In addition to other biomarkers of specificity, the Prostate Health Index has been included to aid biopsy decisions in certain men, given recent FDA approvals.The NCCN Guidelines for Prostate Cancer Early Detection provide recommendations for men choosing to participate in an early detection program for prostate cancer. These NCCN Guidelines Insights highlight notable recent updates. Overall, the 2014 update represents a more streamlined and concise set of recommendations. The panel stratified the age ranges at which initiating testing for prostate cancer should be considered. Indications for biopsy include both a cutpoint and the use of multiple risk variables in combination. In addition to other biomarkers of specificity, the Prostate Health Index has been included to aid biopsy decisions in certain men, given recent FDA approvals.The NCCN Guidelines for Prostate Cancer Early Detection provide recommendations for men choosing to participate in an early detection program for prostate cancer. These NCCN Guidelines Insights highlight notable recent updates. Overall, the 2014 update represents a more streamlined and concise set of recommendations. The panel stratified the age ranges at which initiating testing for prostate cancer should be considered. Indications for biopsy include both a cutpoint and the use of multiple risk variables in combination. In addition to other biomarkers of specificity, the Prostate Health Index has been included to aid biopsy decisions in certain men, given recent FDA approvals.The NCCN Guidelines for Prostate Cancer Early Detection provide recommendations for men choosing to participate in an early detection program for prostate cancer. These NCCN Guidelines Insights highlight notable recent updates. Overall, the 2014 update represents a more streamlined and concise set of recommendations. The panel stratified the age ranges at which initiating testing for prostate cancer should be considered. Indications for biopsy include both a cutpoint and the use of multiple risk variables in combination. In addition to other biomarkers of specificity, the Prostate Health Index has been included to aid biopsy decisions in certain men, given recent FDA approvals. "
    output = validate_query(input)
    print_test_result(input, output)

if __name__ == "__main__":
    # run the test individually
    # asyncio.run(test_generate_keywords())
    # asyncio.run(test_refine_query())
    # asyncio.run(test_expand_keyword())
    # asyncio.run(test_vector_search())
    # test_extract_text_from_document()
    test_validate_query()
    
    # # Run all tests
    # async def run_all_tests():
    #     await test_generate_keywords()
    #     await test_refine_query()
    #     await test_expand_keyword()
    #    await test_vector_search()
    
    # asyncio.run(run_all_tests())