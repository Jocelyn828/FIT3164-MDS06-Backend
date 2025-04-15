import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import ollama
import PyPDF2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prostate_cancer_analysis")

# Create directories for results
Path("./results").mkdir(exist_ok=True)

class LocalPDFAnalyzer:
    def __init__(self, model="gemma:7b"):
        self.model = model
            
        self.exclusion_criteria = [
            "Not Prostate Cancer/ irrelevant/ not related to study objective",
            "Unrelated (Treatment/Management/Diagnosis - Not prostate cancer-specific)",
            "Not by country health authorities or medical organisation",
            "Not research study objectives - ie not prostate cancer-related or decision aid or guideline or programme related to prostate cancer",
            "Intervention/ primary study/erratum/ conference abstract/ commentary/ clinic trial/ newsletter/ magazine/protocol/ literature review",
            "Conhoference abstract",
            "Cost effective/ budget impact",
            "Attitude, belief, experience, behaviour, knowledge",
            "Primary study or not systematic review",
            "News, lecturer note, thesis, draft, proposal, module, stimulation, presscast, fact sheet, opinion, editorial",
            "Non-english",
            "Public consumption /patient info/ patient guide",
            "Conhoference abstract",
        ]

    def extract_text_from_pdf(self, pdf_path):
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                logger.warning(f"Extracted empty text from {pdf_path}")
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def summarize_document(self, text):
        """Summarize document to reduce token count"""
        if not text.strip():
            return ""
            
        try:
            # Get document length to determine summary approach
            word_count = len(text.split())
            if word_count < 1000:
                # For short documents, no need to summarize
                return text[:5000]
                
            # For longer documents, extract key sections and summarize
            prompt = f"""
            Extract and summarize the most important information from this document about prostate cancer research.
            Focus especially on:
            1. The main topic/focus of the document
            2. Research methodology (if applicable)
            3. Key findings or recommendations
            4. Any key terminology that characterizes this document
            
            Keep the summary under 1500 words while preserving the essential information.
            
            Document text:
            {text[:8000]}
            """
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            
            summary = response['message']['content']
            return summary
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            return text[:5000]  # Fallback to truncation
    
    def classify_document(self, text, classification_type="exclusion"):
        if not text.strip():
            return {"classification": "ERROR: Empty text, cannot classify", "keywords": [], "reason": ""}
            
        try:
            if classification_type == "exclusion":
                criteria = self.exclusion_criteria
                prompt = f"""
                You are reviewing medical documents about prostate cancer that have been classified for EXCLUSION.
                Your task is to analyze this document summary and explain WHY it meets exclusion criteria.
                
                Here are the possible exclusion criteria:
                {json.dumps(criteria, indent=2)}
                
                Document summary:
                {text}
                
                Provide your analysis in the following JSON format:
                {{
                  "classification": "The specific exclusion criterion that applies",
                  "keywords": ["list", "of", "key", "terms", "from", "the", "summary", "that", "justify", "exclusion"],
                  "reason": "A detailed explanation of why this document should be excluded based on the criterion"
                }}
                
                Return ONLY the JSON with no additional text.
                """
            else:  # inclusion
                prompt = f"""
                You are reviewing medical documents about prostate cancer that have been classified for INCLUSION.
                Your task is to analyze this document summary and explain WHY it meets inclusion criteria for prostate cancer research.
                
                Document summary:
                {text}
                
                Provide your analysis in the following JSON format:
                {{
                  "classification": "Inclusion reason (brief phrase)",
                  "keywords": ["list", "of", "key", "terms", "from", "the", "summary", "that", "justify", "inclusion"],
                  "reason": "A detailed explanation of why this document should be included in prostate cancer research"
                }}
                
                Return ONLY the JSON with no additional text.
                """
                
            # Call Ollama
            response = ollama.chat(model=self.model, messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ])
            
            result_text = response['message']['content']
            
            # Try to parse JSON from the response
            try:
                # Extract JSON if it's within code blocks
                if "```json" in result_text and "```" in result_text.split("```json")[1]:
                    json_str = result_text.split("```json")[1].split("```")[0]
                    result = json.loads(json_str)
                elif "```" in result_text and "```" in result_text.split("```")[1]:
                    json_str = result_text.split("```")[1].split("```")[0]
                    result = json.loads(json_str)
                else:
                    # Try direct JSON parsing
                    result = json.loads(result_text)
                
                return result
            except json.JSONDecodeError:
                # Fallback - return as text if JSON parsing fails
                logger.warning(f"Failed to parse JSON classification: {result_text}")
                return {
                    "classification": classification_type,
                    "keywords": [],
                    "reason": result_text
                }
                
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return {
                "classification": "ERROR",
                "keywords": [],
                "reason": f"Classification failed: {str(e)}"
            }
    
    def extract_patterns(self, classification_results):
        if not classification_results["exclusion_results"] and not classification_results["inclusion_results"]:
            logger.warning("No classification results to extract patterns from")
            return {
                "exclusion_patterns": [],
                "inclusion_patterns": []
            }
        
        exclusion_data = [{
            "file": r["file"],
            "classification": r["classification"]["classification"] if isinstance(r["classification"], dict) else r["classification"],
            "keywords": r["classification"]["keywords"] if isinstance(r["classification"], dict) and "keywords" in r["classification"] else [],
            "reason": r["classification"]["reason"] if isinstance(r["classification"], dict) and "reason" in r["classification"] else ""
        } for r in classification_results["exclusion_results"]]
        
        inclusion_data = [{
            "file": r["file"],
            "classification": r["classification"]["classification"] if isinstance(r["classification"], dict) else r["classification"],
            "keywords": r["classification"]["keywords"] if isinstance(r["classification"], dict) and "keywords" in r["classification"] else [],
            "reason": r["classification"]["reason"] if isinstance(r["classification"], dict) and "reason" in r["classification"] else ""
        } for r in classification_results["inclusion_results"]]
        
        prompt = f"""
        You are analyzing document classification results for prostate cancer research papers.
        
        ANALYZE THE EXCLUSION AND INCLUSION DOCUMENTS BELOW:
        
        EXCLUSION DOCUMENTS:
        {json.dumps(exclusion_data, indent=2)}
        
        INCLUSION DOCUMENTS:
        {json.dumps(inclusion_data, indent=2)}
        
        INSTRUCTIONS:
        1. Analyze ONLY the 'classification', 'keywords', and 'reason' fields for these documents
        2. Identify the most common patterns or themes for why documents were excluded or included
        3. Base your patterns STRICTLY on the evidence in these fields - do not invent new reasons
        4. For each pattern, list representative keywords from the actual documents
        
        Format your response as JSON with this simpler structure:
        {{
        "exclusion_patterns": [
            {{
            "pattern": "Name of pattern/theme",
            "keywords": ["representative", "keywords", "from", "documents"],
            "evidence": "Brief explanation with examples from specific documents"
            }},
            ...
        ],
        "inclusion_patterns": [
            {{
            "pattern": "Name of pattern/theme", 
            "keywords": ["representative", "keywords", "from", "documents"],
            "evidence": "Brief explanation with examples from specific documents"
            }},
            ...
        ]
        }}
        
        Return ONLY the JSON with no additional text.
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
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
                
            logger.warning("Could not parse JSON from response, returning raw text")
            return {
                "error": "Failed to parse JSON response",
                "raw_response": content
            }
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return {
                "error": str(e),
                "exclusion_patterns": [],
                "inclusion_patterns": []
            }
    
    def process_pdf_files(self, exclusion_pdfs, inclusion_pdfs):
        exclusion_results = []
        inclusion_results = []
        
        # Process exclusion PDFs
        logger.info(f"Processing {len(exclusion_pdfs)} exclusion PDFs...")
        for pdf_path in tqdm(exclusion_pdfs):
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if text.strip():
                # First summarize if needed
                if len(text.split()) > 1000:
                    summary = self.summarize_document(text)
                else:
                    summary = text[:5000]  # Use truncated text for shorter documents
                    
                # Get classification
                classification = self.classify_document(summary, "exclusion")
                result = {
                    "file": os.path.basename(pdf_path),
                    "summary": summary,  # Include the summary used for classification
                    "classification": classification
                }
                exclusion_results.append(result)
                logger.info(f"Classified exclusion document: {os.path.basename(pdf_path)}")
            else:
                logger.error(f"Failed to extract text from: {pdf_path}")
        
        # Process inclusion PDFs
        logger.info(f"Processing {len(inclusion_pdfs)} inclusion PDFs...")
        for pdf_path in tqdm(inclusion_pdfs):
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if text.strip():
                # First summarize if needed
                if len(text.split()) > 1000:
                    summary = self.summarize_document(text)
                else:
                    summary = text[:5000]  # Use truncated text for shorter documents
                    
                # Get classification
                classification = self.classify_document(summary, "inclusion")
                result = {
                    "file": os.path.basename(pdf_path),
                    "summary": summary,  # Include the summary used for classification
                    "classification": classification
                }
                inclusion_results.append(result)
                logger.info(f"Classified inclusion document: {os.path.basename(pdf_path)}")
            else:
                logger.error(f"Failed to extract text from: {pdf_path}")
        
        # Extract patterns only if we have results
        classification_results = {
            "exclusion_results": exclusion_results,
            "inclusion_results": inclusion_results
        }
        
        patterns = self.extract_patterns(classification_results)
        
        # Save results
        with open("./results/exclusion_results.json", "w") as f:
            json.dump(exclusion_results, f, indent=2)
            
        with open("./results/inclusion_results.json", "w") as f:
            json.dump(inclusion_results, f, indent=2)
            
        with open("./results/patterns.json", "w") as f:
            json.dump(patterns, f, indent=2)
        
        return {
            "exclusion_results": exclusion_results,
            "inclusion_results": inclusion_results,
            "patterns": patterns
        }


if __name__ == "__main__":
    analyzer = LocalPDFAnalyzer(model="gemma:7b")
    
    L1_exclusion_folder ="./L1_exclusion_pdfs"
    L2_exclusion_folder ="./L2_exclusion_pdfs"
    inclusion_folder ="./inclusion_pdfs"
    
    # Get Level 1 Excluded PDFs
    L1_exclusion = [os.path.join(L1_exclusion_folder, f) for f in os.listdir(L1_exclusion_folder) if f.endswith(".pdf")]

    # Get Level 2 Excluded PDFs
    L2_exclusion = [os.path.join(L2_exclusion_folder, f) for f in os.listdir(L2_exclusion_folder) if f.endswith(".pdf")]
    exclusion_pdfs = L1_exclusion + L2_exclusion

    # Get 10 Included PDFs
    inclusion_pdfs = [os.path.join(inclusion_folder, f) for f in os.listdir(inclusion_folder) if f.endswith(".pdf")]

    
    # Run the analysis
    results = analyzer.process_pdf_files(exclusion_pdfs, inclusion_pdfs)
    
    # Print summary
    print("\nAnalysis Complete!")
    print(f"Processed {len(results['exclusion_results'])} exclusion documents")
    print(f"Processed {len(results['inclusion_results'])} inclusion documents")
    
    # Print pattern summary if available
    if results['patterns'] and 'exclusion_patterns' in results['patterns'] and 'inclusion_patterns' in results['patterns']:
        print("\nKey Exclusion Patterns:")
        for pattern in results['patterns'].get('exclusion_patterns', [])[:5]:
            print(f"- {pattern['pattern']}: {pattern['description']}")
            print(f"  Example keywords: {', '.join(pattern['example_keywords'][:5])}")
        
        print("\nKey Inclusion Patterns:")
        for pattern in results['patterns'].get('inclusion_patterns', [])[:5]:
            print(f"- {pattern['pattern']}: {pattern['description']}")
            print(f"  Example keywords: {', '.join(pattern['example_keywords'][:5])}")
    
    print("\nDetailed results saved to the 'results' directory")