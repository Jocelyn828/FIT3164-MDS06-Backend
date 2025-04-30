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