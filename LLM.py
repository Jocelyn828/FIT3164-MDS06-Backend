from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import json
from prompts import *

# Define output structure for different prompt lengths
class ShortResearchQuery(BaseModel):
    suggested_subtopics: List[str] = Field(..., description="Suggested possible subtopics if the topic is broad")
    subtopics_reason: str = Field(..., description="Explain the reasoning behind the suggested subtopics")

class MediumResearchQuery(BaseModel):
    initial_query: str = Field(..., description="Initial search query generated from the topic")
    refinement_options: List[str] = Field(..., description="Possible ways to refine the query")
    refined_query: str = Field(..., description="Refined search query with improved terms")
    key_concepts: List[str] = Field(..., description="Important concepts identified in the topic")
    refinement_reason: str = Field(..., description="Reasoning behind query refinement")

class LongResearchQuery(BaseModel):
    validation_checklist: List[str] = Field(..., description="Checklist to validate the input scope, timeframe, and sources")
    initial_query: str = Field(..., description="Initial search query generated from the topic")
    refinement_suggestions: List[str] = Field(..., description="Suggested ways to improve the query")
    refined_query: str = Field(..., description="Refined search query with improved terms")
    key_concepts: List[str] = Field(..., description="Important concepts identified in the topic")
    refinement_reason: str = Field(..., description="Reasoning behind query refinement")

# Unified output model
class ResearchOutput(BaseModel):
    short_result: Optional[ShortResearchQuery] = None
    medium_result: Optional[MediumResearchQuery] = None
    long_result: Optional[LongResearchQuery] = None

# Determine template based on input length
def select_template(topic: str):
    words = len(topic.split())
    if words <= 3:
        return short_template, ShortResearchQuery
    elif words <= 10:
        return medium_template, MediumResearchQuery
    else:
        return long_template, LongResearchQuery

async def main():
    topic = "LLM"
    template, outputModel = select_template(topic)
    prompt = ChatPromptTemplate.from_template(template)
    # Small Model
    # model = ChatOllama(**{'model': 'llama3.2:3b', 'temperature': 0.3, 'seed': 42})
    # Middle Model
    model = ChatOllama(**{'model': 'deepseek-r1:14b', 'temperature': 0.3, 'seed': 42})

    structured_llm = model.with_structured_output(outputModel, method="json_schema")
    
    chain = prompt | structured_llm
    
    task = asyncio.create_task(
        chain.ainvoke({"topic": topic})
    )
    
    try:
        await asyncio.wait_for(task, 60.0)
        # await task
    except asyncio.TimeoutError:
        print("Timeout occurred")
    
    result = task.result().model_dump()
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())