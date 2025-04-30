import asyncio
import json
import ollama
        
async def generate_keywords(input):
    prompt = f"""
    You are an intelligent assistant that helps users extract important keywords from scientific or technical abstracts to assist in forming effective search queries.
    You will receive an abstract from a research article.

    Abstract: {input}

    Task:
    1. Carefully read the provided abstract.
    2. Identify and extract 7 to 12 meaningful keywords or key phrases that best represent the main ideas, topics, and important concepts of the abstract.
    3. Keywords should be specific (not too broad) and relevant to the content, be short and concise. 
    4. Avoid generic words like "study," "research," "article," "paper."
    5. Prefer nouns and noun phrases (e.g., "machine learning algorithm," "gene mutation," "urban heat island effect").
    6. Do not invent information not found in the abstract.

    Format your response as JSON with this structure:
    {{
    "keywords": [
        "representative", 
        "keywords", 
        "from", 
        "abstract"
    ]
    }}

    Return ONLY the JSON with no additional text.
    """
    try:
        response = ollama.chat(model="deepseek-r1:14b", messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])
        
        # Simplified JSON extraction
        content = response['message']['content']
        
        # Try to extract JSON using a more robust approach
        import re
        
        # First try to find JSON between code blocks
        if "```json" in content:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                return json.loads(json_match.group(1))
        elif "```" in content:
            json_match = re.search(r'```\s*([\s\S]*?)\s*```', content)
            if json_match:
                return json.loads(json_match.group(1))
        
        # Then try to find any JSON object
        json_match = re.search(r'(\{[\s\S]*\})', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
            
        return {
            "error": "Failed to parse JSON response",
            "raw_response": content
        }
    except Exception as e:
        return {
            "error": str(e),
            "keywords": []
        }
    
async def generate_query(input, keywords):
    # Prompt with keywords
    prompt = f"""
    You are an intelligent assistant that helps users draft an initial search query intended for semantic search using embeddings.
    You are given:
    - An abstract of a research article: {input}
    - A list of user-selected keywords: {keywords}
    At this stage, you must only create a focused, positive initial query, without applying any exclusion criteria yet.

    Task:
    1. Read the abstract carefully.
    2. Understand the key topic, research problem, and context.
    3. Using the provided keywords, generate an initial search query that captures the article’s main ideas.
    4. Ensure the query logically integrates most of the selected keywords.
    5. Do not add negative filters (no NOT, no exclusions).

    Format your response as JSON with this structure:
    {{
    "inital_query": "Name of pattern/theme"
    }}

    Return ONLY the JSON with no additional text.
    """

    # Prompt without keywords
    prompt_no = f"""
    You are an intelligent assistant that helps users draft an initial search query intended for semantic search using embeddings.
    You are given:
    - An abstract of a research article: {input}

    At this stage, you must only create a focused, positive initial query, without applying any exclusion criteria.

    Task:
    1. Read the abstract carefully.
    2. Understand the key topic, research problem, and context.
    3. Generate an initial search query that captures the article’s main ideas.
    4. Do not add negative filters (no NOT, no exclusions).

    Format your response as JSON with this structure:
    {{
    "inital_query": "Name of pattern/theme"
    }}

    Return ONLY the JSON with no additional text.
    """

    try:
        response = ollama.chat(model="deepseek-r1:14b", messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])
        
        # Simplified JSON extraction
        content = response['message']['content']
        
        # Try to extract JSON using a more robust approach
        import re
        
        # First try to find JSON between code blocks
        if "```json" in content:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                return json.loads(json_match.group(1))
        elif "```" in content:
            json_match = re.search(r'```\s*([\s\S]*?)\s*```', content)
            if json_match:
                return json.loads(json_match.group(1))
        
        # Then try to find any JSON object
        json_match = re.search(r'(\{[\s\S]*\})', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
            
        return {
            "error": "Failed to parse JSON response",
            "raw_response": content
        }
    except Exception as e:
        return {
            "error": str(e),
            "initial_query": ""
        }

async def main():
    # abstract from inclusion pdf article_10
    input = """
    Background: The recommendations and the updated EAU guidelines consider early
    detection of PCa with the purpose of reducing PCa-related mortality and the develop-
    ment of advanced or metastatic disease.
    Objective: This paper presents the recommendations of the European Association of
    Urology (EAU) for early detection of prostate cancer (PCa) in men without evidence of
    PCa-related symptoms.
    Evidence acquisition: The working panel conducted a systematic literature review and
    meta-analysis of prospective and retrospective clinical studies on baseline prostate-
    specific antigen (PSA) and early detection of PCa and on PCa screening published
    between 1990 and 2013 using Cochrane Reviews, Embase, and Medline search
    strategies.
    Evidence synthesis: The level of evidence and grade of recommendation were analysed
    according to the principles of evidence-based medicine. The current strategy of the EAU
    recommends that (1) early detection of PCa reduces PCa-related mortality; (2) early
    detection of PCa reduces the risk of being diagnosed and developing advanced and
    metastatic PCa; (3) a baseline serum PSA level should be obtained at 40–45 yr of age; (4)
    intervals for early detection of PCa should be adapted to the baseline PSA serum
    concentration; (5) early detection should be offered to men with a life expectancy
    10 yr; and (6) in the future, multivariable clinical risk-prediction tools need to be
    integrated into the decision-making process.
    Conclusions: A baseline serum PSA should be offered to all men 40–45 yr of age to
    initiate a risk-adapted follow-up approach with the purpose of reducing PCa mortality
    and the incidence of advanced and metastatic PCa. In the future, the development and
    application of multivariable risk-prediction tools will be necessary to prevent over
    diagnosis and over treatment.
    """

    # abstract from inclusion pdf article_11
    input = """
    The NCCN Guidelines for Prostate Cancer Early Detection provide recommendations for men choosing to participate in an early de-
    tection program for prostate cancer. These NCCN Guidelines Insights highlight notable recent updates. Overall, the 2014 update
    represents a more streamlined and concise set of recommendations. The panel stratified the age ranges at which initiating testing for
    prostate cancer should be considered. Indications for biopsy include both a cutpoint and the use of multiple risk variables in combina-
    tion. In addition to other biomarkers of specificity, the Prostate Health Index has been included to aid biopsy decisions in certain men,
    given recent FDA approvals.
    """
    
    keywords = await generate_keywords(input)
    print("Keywords:")
    for keyword in keywords['keywords']:
        print(f"-  {keyword}")

    # Example - User selects the first 5 keywords
    keywords = keywords['keywords'][:5]
    
    result = await generate_query(input, keywords)
    print("Initial Query:")
    print(result['initial_query'])

if __name__ == "__main__":
    asyncio.run(main())