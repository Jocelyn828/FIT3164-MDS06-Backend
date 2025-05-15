# Zero-shot Prompt
zero_template = """You are a research assistant. Generate an academic search query based on the topic below and refine it.

Topic: {topic}

Instructions:
1. Analyze the topic and identify key concepts
2. Create a initial search string using key terms
3. Refine the query by:
   - Expand by adding synonyms, technical terms, and controlled vocabulary
   - Including technical jargon from the field
   - Applying search operators (e.g., OR, wildcards)
4. Explain your refinement process

Format your response with:
- Initial query
- Refined query
- Key concepts identified
- Refinement reasoning"""


# One-shot Prompt
one_template = """You are a research assistant. Generate an academic search query based on the topic below and refine it.

Topic: {topic}

Instructions:
1. Analyze the topic and identify key concepts
2. Create a initial search string using key terms
3. Refine the query by:
   - Expand by adding synonyms, technical terms, and controlled vocabulary
   - Including technical jargon from the field
   - Applying search operators (e.g., OR, wildcards)
4. Explain your refinement process

Example:
Topic: "I need research papers about AI in medicine"
"initial_query" "AI in medicine"  
"refined_query": "Artificial Intelligence applications in medical diagnostics AND treatment"

Format your response with:
- Initial query
- Refined query
- Key concepts identified
- Refinement reasoning"""

# Few-shot Prompt
few_template = """You are a research assistant. Generate an academic search query based on the topic below and refine it.

Topic: {topic}

Instructions:
1. Analyze the topic and identify key concepts
2. Create a initial search string using key terms
3. Refine the query by:
   - Expand by adding synonyms, technical terms, and controlled vocabulary
   - Including technical jargon from the field
   - Applying search operators (e.g., OR, wildcards)
4. Explain your refinement process

Example:
Topic: "I need research papers about AI in medicine"
"initial_query" "AI in medicine"  
"refined_query": "Artificial Intelligence applications in medical diagnostics AND treatment"

Example:
Topic: "online learning papers"
"initial_query" "online learning"  
"refined_query": "online learning AND (education OR e-learning) AND (student engagement OR challenges)"

Format your response with:
- Initial query
- Refined query
- Key concepts identified
- Refinement reasoning"""


# New prompt for keyword expansion
keyword_expansion_template = """You are an academic search assistant that helps users formulate effective search queries.
Given a simple keyword or short phrase, expand it into a well-formulated search query that would retrieve relevant academic papers.

Keyword: {keyword}

Instructions:
1. Transform this keyword into a complete, natural language search query
2. Include relevant terminology and concepts related to the keyword
3. Make sure the expanded query is clear, specific, and academically oriented
4. Keep the tone scholarly and the language precise
5. The result should be 1-2 sentences max, focused on the core concept

Your expanded query:"""


# New prompt for keyword expansion
keyword_expansion_template = """You are an academic search assistant that helps users formulate effective search queries.
Given a simple keyword or short phrase, expand it into a well-formulated search query that would retrieve relevant academic papers.

Keyword: {keyword}

Instructions:
1. Transform this keyword into a complete, natural language search query
2. Include relevant terminology and concepts related to the keyword
3. Make sure the expanded query is clear, specific, and academically oriented
4. Keep the tone scholarly and the language precise
5. The result should be 1-2 sentences max, focused on the core concept

Your expanded query:"""


# New prompt for query refinement
combined_query_template = """
You are a research query refinement assistant. 
Given a search query that includes boolean operators (AND/OR), 
create a more effective search query for finding academic articles.

Combined Query: {combined_query}

Please provide:
1. A refined search query that will be more effective for finding relevant articles
2. The key concepts identified in the query
3. Your reasoning for how you refined the query

Format your response as a JSON object with the following structure:
{
    "refined_query": "your refined query",
    "key_concepts": ["concept1", "concept2", "..."],
    "refinement_reason": "explanation of your refinement approach"
}
"""


keyword_generation_template = """
You are an AI assistant specializing in medical research and information retrieval. Your task is to analyze a search query and extract key concepts or keywords that would help refine a medical literature search.

For the query: "{query}"

Extract 8-12 distinct keywords or key phrases that:
1. Capture the essential medical or research concepts in the query
2. Include specific technical terms, conditions, treatments, or methodologies
3. Represent different aspects or dimensions of the query (diagnosis, treatment, etiology, epidemiology, etc.)
4. Vary in specificity (some broader terms, some more specific)

For example, if the query is "diabetes treatment":
- Glycemic control
- Insulin therapy
- Oral antidiabetic agents
- HbA1c management
- Diabetic complications
- Metformin
- Type 2 diabetes
- Pancreatic beta cells
- GLP-1 receptor agonists
- Diabetic nephropathy
- Incretin mimetics

Format your response as a list of medical keywords or key phrases. Each keyword should be concise (1-3 words typically) and highly relevant to the query. Include specific biomarkers, procedures, anatomical structures, cell types, mechanisms, or methodologies when relevant.
"""