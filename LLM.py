from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import asyncio
import json
from prompts import *
import os
import fitz
from PyPDF2 import PdfReader
import pandas as pd

# Define output structure for query refinement
class ResearchQuery(BaseModel):
    initial_query: str = Field(..., description="The given query")
    refined_query: str = Field(..., description="Refined search query with improved terms")
    refinement_reason: str = Field(..., description="Reasoning behind query refinement")

class ResearchOutput(BaseModel):
    result: ResearchQuery

# Load dataset
df = pd.read_excel("Evaluation_Dataset.xlsx")

async def main():
    query = """
Early detection of prostate cancer is a critical aspect of disease management, involving prostate-specific antigen (PSA) screening, risk-adapted strategies, and evolving clinical guidelines. Various national and international health organizations have established recommendations to balance early detection with concerns about overdiagnosis and overtreatment. PSA-based screening remains the primary diagnostic tool, supplemented by digital rectal examinations and emerging biomarkers such as PSA derivatives and genetic risk assessments. Recent advancements in prostate cancer risk models, biomarker research, and imaging techniques aim to improve screening accuracy while minimizing unnecessary biopsies.\
Decision aids and shared decision-making frameworks are increasingly emphasized to ensure patients receive personalized, evidence-based guidance. Some guidelines recommend against routine screening for asymptomatic men due to the potential risks, while others propose targeted screening approaches based on individual risk factors such as age, family history, and genetic predisposition. Research also explores the effectiveness of decision aids in helping patients weigh the benefits and risks of screening.\
This review synthesizes key findings from prostate cancer screening guidelines, biomarker innovations, and systematic reviews, highlighting the ongoing challenges in optimizing early detection strategies. Future directions include refining screening protocols, integrating artificial intelligence in diagnostics, and enhancing patient-centered approaches to prostate cancer detection.
"""  
    # query = "prostate cancer early detection"
    template = query_refine_template
 
    include_count = 0
    exclude_count = 0
        
    file_name = 'Evaluation Dataset.json'

    # Read the JSON file with UTF-8 encoding
    with open(file_name, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # Create individual EvaluationArticle objects for each entry
    for article_data in data:
        
        # Original Article fields
        final_level_1_consensus=article_data.get('Final Level 1 Consensus', '').lower().strip()
        exclusion_reason_final_level_1=article_data.get('Exclusion Reason Final Level 1', '')
        final_level_2_consensus=article_data.get('Final Level 2 Consensus', '').lower().strip()
        exclusion_reason_final_level_2=article_data.get('Exclusion Reason Final Level 2', '')
        title=article_data.get('Title', '')
        theme=article_data.get('Theme', '')
        abstract=article_data.get('Abstract', '')

        if final_level_2_consensus == "include":
            template += f"\nTitle: {title}\nAbstract: {abstract}\nDecision: INCLUDE\n"
            include_count += 1
        else:
            if final_level_1_consensus == "exclude":
                template += f"\nTitle: {title}\nAbstract: {abstract}\nDecision: EXCLUDE ({exclusion_reason_final_level_1})\n"
                
            else:
                template += f"\nTitle: {title}\nAbstract: {abstract}\nDecision: EXCLUDE ({exclusion_reason_final_level_2})\n"
            exclude_count += 1

    template += """Analyze the given papers to extract key concepts, and refine the initial search query: '{query}'. Explain the refinement reasoning.
    Refine the query by:
        - Identify essential keywords, expand with synonyms, technical terms, and controlled vocabulary
        - Apply Boolean operators (AND, OR, NOT) to include/exclude terms
        - Incorporate field-specific terminology
        - Apply wildcards, phrase searching, and proximity operators for precision
    """

    # Setup LLM model
    prompt = ChatPromptTemplate.from_template(template)
    # print(prompt)
    model = ChatOllama(**{'model': 'deepseek-r1:14b', 'temperature': 0.2, 'seed': 42})
    structured_llm = model.with_structured_output(ResearchOutput, method="json_schema")
    
    chain = prompt | structured_llm

    # Run LLM query asynchronously
    task = asyncio.create_task(chain.ainvoke({"query": query}))

    try:
        await asyncio.wait_for(task, 60.0)
    except asyncio.TimeoutError:
        print("Timeout occurred")

    result = task.result().model_dump()
    print(json.dumps(result, indent=2))
    print(f"Include Count: {include_count}")
    print(f"Exclude Count: {exclude_count}")

if __name__ == "__main__":

    asyncio.run(main())
