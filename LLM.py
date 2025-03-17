from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import asyncio
import json
from prompts import zero_template, one_template, few_template

# Define output structure
class ResearchQuery(BaseModel):
    initial_query: str = Field(..., description="Initial search query generated from the topic")
    refined_query: str = Field(..., description="Refined search query with improved terms")
    key_concepts: List[str] = Field(..., description="Important concepts identified in the topic")
    refinement_reason: str = Field(..., description="Reasoning behind query refinement")

class ResearchOutput(BaseModel):
    result: ResearchQuery

template = zero_template

async def main():
    prompt = ChatPromptTemplate.from_template(template)
    # Small Model
    model = ChatOllama(**{'model': 'llama3.2:3b', 'temperature': 0.2, 'seed': 42})
    # Middle Model
    # model = ChatOllama(**{'model': 'deepseek-r1:14b', 'temperature': 0.2, 'seed': 42})

    structured_llm = model.with_structured_output(ResearchOutput, method="json_schema")
    
    chain = prompt | structured_llm
    
    # Example topic - replace with user input
    task = asyncio.create_task(
        chain.ainvoke({"topic": "I need papers about LLM"})
    )
    
    try:
        await asyncio.wait_for(task, 60.0)
        # await task
    except asyncio.TimeoutError:
        print("Timeout occurred")
    
    result = task.result().model_dump()
    # print(json.dumps(result['result']['initial_query'], indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())