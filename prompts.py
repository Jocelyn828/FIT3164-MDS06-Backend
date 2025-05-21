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
refine_query_template = """
## Prompt Description
You are an expert assistant helping to refine and optimize a verbose or messy search query for use in **semantic document retrieval** using embedding-based models (e.g., SBERT, OpenAI, Cohere).

---

## Initial Query
```text
{text}
````

---

## Task Instructions

1. Carefully read and understand the initial query to identify the **main topic**, **supporting details**, and the **underlying information need**.
2. Rephrase the query into a **natural, focused, and concise sentence** that best represents the user's intent for document retrieval — avoid Boolean logic or keywords unless they clarify meaning.
3. Extract 4–8 **key concepts or phrases** from the original query that are central to the search intent. These should be semantically meaningful terms, not formatting instructions or metadata.
4. Write a brief explanation describing:

   * What the original query was about
   * How the refined version improves clarity for semantic understanding
   * Any generalizations or assumptions made (e.g., excluding non-English based on examples)

---

## Response Format

```json
{{
  "Initial query": "The original verbose or unstructured input query",
  "Refined Query": "A clean, focused semantic search query in natural language",
  "key concepts": ["important", "topics", "and", "phrases", "from", "the", "query"],
  "reason": "Explanation of what the user was looking for and how the new query expresses that more clearly for an embedding-based system."
}}
```

---

## Example 1

**Initial query:**

```
("prostate-specific antigen" OR PSA OR "screening guidelines" OR "risk factors" OR "genetic variants" OR "machine learning" OR "support vector machines") 
NOT ("Chinese" OR "French" OR "general AI applications" OR "unrelated medical field")
```

**Expected output:**

```json
{{
  "Initial query": "('prostate-specific antigen' OR PSA OR ...)",
  "Refined Query": "Studies using machine learning models like SVMs to analyze PSA levels or genetic risk factors for prostate cancer screening, focused on English-language publications.",
  "key concepts": ["prostate-specific antigen", "PSA", "machine learning", "support vector machines", "genetic risk factors", "prostate cancer screening", "English-language filter"],
  "reason": "The original query aimed to find research on machine learning approaches to PSA and genetic factors in prostate cancer screening, while excluding irrelevant topics and non-English documents. The refined query summarizes that intent naturally and is better suited for embedding-based search."
}}
```

---

## Final Note

Return **only** the JSON with **no extra explanation or comments**.
"""

boolean_refine_query_template = """
## Prompt Description
You are an expert assistant helping to refine and optimize a verbose or messy search query for use in **semantic document retrieval** using **embedding-based models**. The goal is to clean up the query, group concepts using Boolean logic (AND, OR, NOT) for clarity, and extract key ideas — while keeping the query semantically understandable as natural language.

---

## Initial Query
```text
{text}
````

---

## Task Instructions

1. Carefully read and understand the initial query to identify the **main topic**, **supporting details**, and the **underlying information need**.
2. Reconstruct the query using Boolean operators (`AND`, `OR`, `NOT`) to logically organize ideas and improve retrieval precision and clarity.
3. Extract 4–8 **key concepts or phrases** that are central to the query’s intent. These should be meaningful terms that reflect the user's search goals.
4. Generalize any field-based or literal exclusions (e.g., `title:`, `author:`, direct document titles). Instead of copying such syntax, abstract their **semantic meaning**.

   * For example, if a user excludes a specific article by title, infer what **topic or theme** that article represents (e.g., "epidemiology", "historical guidelines") and exclude it at a conceptual level.
5. Write a short explanation describing:

   * What the query is about
   * How the refined version improves semantic clarity
   * Any exclusions or generalizations applied

---

## Response Format

```json
{{
  "Initial query": "The original unstructured input query",
  "Refined query": "A Boolean-style, semantically structured query suitable for academic or embedding-based search",
  "Key concepts": ["concept 1", "concept 2", "concept 3"],
  "Reason": "A concise explanation of how the refined query captures the core intent, improves clarity, and organizes key ideas using Boolean logic — avoiding literal field-based syntax where possible."
}}
```

---

## Final Note

Return **only** the JSON output. Do not include markdown, extra commentary, or raw field-level metadata such as `title:` or citation strings.
"""


keyword_generation_template = """
You are an AI assistant specializing in medical research and information retrieval. Your task is to analyze a search query and extract key concepts or keywords that would help refine a medical literature search.

For the query: "{query}"

Extract 6-8 distinct keywords or key phrases that:
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


