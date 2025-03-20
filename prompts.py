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



short_template = """You are a research assistant. Your task is to generate an academic search query based on a given topic. If the topic is ambiguous, list distinct meanings from completely different domains

Topic: {topic}

Instructions:
1. Identify possible interpretations of the topic.  
2. List 3-5 distinct meanings of the topic across domains if it is ambiguous.
3. Explain how these interpretations can guide the search process.

Response Format:
- Suggest distinct interpretations for narrowing
- Explanation of how these subtopics can help user focus their search

Example:
Topic: "apple"
json output:
{{
   "suggested_subtopics": ["fruit", "technology company", "health benefits"],
   "subtopics_reason": "apple has multiple meanings, including a fruit, a technology company, and a symbol of health benefits. By identifying these distinct interpretations, users can narrow down their search focus."
}} 


Example:
Topic: "cloud"
json output:
{{
   "suggested_subtopics": ["weather", "meteorology", "cloud computing", "storage"],
   "subtopics_reason": "cloud can refer to atmospheric phenomena, cloud computing, or data storage. By clarifying these meanings, users can refine their search to a specific domain."

}} 
"""

medium_template = """You are a research assistant. Generate and refine an academic search query based on the topic below.

Topic: {topic}

Instructions:
1. Analyze the user input to determine the research focus.
2. Identify key concepts and create an initial search query.
3. Offer 2-3 ways to refine the query (e.g., adding methodologies, timeframes, comparison terms).
4. Use search operators (e.g., OR, wildcards) to optimize retrieval.
5. Explain your refinement process.

Response Format:
- Initial query
- Refinement options
- Refined query
- Key concepts
- Refinement reasoning
"""

long_template = """You are a research assistant. Generate and refine an academic search query based on the topic below.

Topic: {topic}

Instructions:
1. Validate the input by checking for:
   - Scope (e.g., specific area within the field)
   - Timeframe (e.g., last five years)
   - Source type (e.g., peer-reviewed journals)
2. Ensure completeness: Are there missing constraints (e.g., methodology, comparison factors)?
3. Generate a precise search query based on the input.
4. Refine the query using synonyms, advanced search operators, and technical terms.
5. Explain your refinement process.

Response Format:
- Validation checklist (scope, timeframe, sources)
- Initial query
- Refinement suggestions
- Refined query
- Key concepts
- Refinement reasoning
"""