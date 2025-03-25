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
    def __init__(self, model="deepseek-r1:1.5b"):
        self.model = model
            
        self.exclusion_criteria = [
            "Not Prostate Cancer/ irrelevant/ not related to study objective",
            "Treatment/ management/ diagnosis",
            "Not by country health authorities or medical organisation",
            "Not research study objectives - ie not prostate cancer-related or decision aid or guideline or programme related to prostate cancer",
            "Management and/or treatment"
        ]
        
        self.inclusion_criteria = [
            "Clinical trials with human subjects",
            "Studies reporting treatment outcomes",
            "Research on prostate cancer biomarkers",
            "Meta-analyses of prostate cancer interventions",
            "Studies on quality of life in prostate cancer patients",
            "Research on prostate cancer screening methods"
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
    
    def classify_document(self, text, classification_type="exclusion"):
        if not text.strip():
            return "ERROR: Empty text, cannot classify"
            
        try:
            if classification_type == "exclusion":
                criteria = self.exclusion_criteria
                prompt = f"""
                You are reviewing medical documents about prostate cancer. 
                Analyze the following document and determine if it should be EXCLUDED based on these criteria:
                {json.dumps(criteria, indent=2)}
                
                If it should be excluded, explain which specific criterion it meets and why.
                If it should NOT be excluded, simply state "INCLUDE: Document does not meet any exclusion criteria."
                
                Document text:
                {text[:10000]}  # Limit text length to avoid token limits
                
                Your analysis:
                """
            else:  # inclusion
                criteria = self.inclusion_criteria
                prompt = f"""
                You are reviewing medical documents about prostate cancer.
                Analyze the following document and determine if it meets these INCLUSION criteria:
                {json.dumps(criteria, indent=2)}
                
                If it meets any inclusion criteria, specify which criteria and explain why.
                If it does NOT meet any inclusion criteria, state "EXCLUDE: Document does not meet any inclusion criteria."
                
                Document text:
                {text[:10000]}  # Limit text length to avoid token limits
                
                Your analysis:
                """
                
            # Call Ollama
            response = ollama.chat(model=self.model, messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ])
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return "ERROR: Classification failed"
    
    def extract_patterns(self, classification_results):
        # Check if we have any results to analyze
        if not classification_results["exclusion_results"] and not classification_results["inclusion_results"]:
            logger.warning("No classification results to extract patterns from")
            return {
                "exclusion_patterns": [],
                "inclusion_patterns": []
            }
        
        prompt = f"""
        You are analyzing a set of document classification results for prostate cancer research papers.
        Based on the following classification results, identify common patterns, keywords, and phrases that 
        characterize documents that should be included or excluded.
        
        Classification results:
        {json.dumps(classification_results, indent=2)}
        
        Extract and list:
        1. Key exclusion patterns (terms, phrases, topics that indicate a document should be excluded)
        2. Key inclusion patterns (terms, phrases, topics that indicate a document should be included)
        
        Format your response as JSON strictly following this structure:
        {{
          "exclusion_patterns": [
            {{
              "term": "specific pattern term",
              "description": "explanation of why this indicates exclusion"
            }},
            ...
          ],
          "inclusion_patterns": [
            {{
              "term": "specific pattern term",
              "description": "explanation of why this indicates inclusion"
            }},
            ...
          ]
        }}
        
        Ensure there is no overlap between exclusion and inclusion patterns.
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ])
            
            # Try to parse JSON from the response
            content = response['message']['content']
            # Find JSON content between ```json and ``` if present
            if "```json" in content and "```" in content.split("```json")[1]:
                json_str = content.split("```json")[1].split("```")[0]
                return json.loads(json_str)
            elif "```" in content and "```" in content.split("```")[1]:
                json_str = content.split("```")[1].split("```")[0]
                return json.loads(json_str)
            
            # Try to find any JSON-like content
            import re
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
                
            logger.warning("Could not parse JSON from response, returning raw text")
            return {"response": content}
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return {"error": str(e)}
    
    def process_pdf_files(self, exclusion_pdfs, inclusion_pdfs):
        exclusion_results = []
        inclusion_results = []
        
        # Process exclusion PDFs
        logger.info(f"Processing {len(exclusion_pdfs)} exclusion PDFs...")
        for pdf_path in tqdm(exclusion_pdfs):
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if text.strip():
                # Get classification
                classification = self.classify_document(text, "exclusion")
                result = {
                    "file": os.path.basename(pdf_path),
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
                # Get classification
                classification = self.classify_document(text, "inclusion")
                result = {
                    "file": os.path.basename(pdf_path),
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
    analyzer = LocalPDFAnalyzer(model="deepseek-r1:1.5b")
    
    L1_exclusion_folder ="./L1_exclusion_pdfs"
    L2_exclusion_folder ="./L2_exclusion_pdfs"
    inclusion_folder ="./inclusion_pdfs"
    
    # Get 10 Level 1 Excluded PDFs
    L1_exclusion = [os.path.join(L1_exclusion_folder, f) for f in os.listdir(L1_exclusion_folder) if f.endswith(".pdf")]

    # Get 10 Level 2 Excluded PDFs
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
            print(f"- {pattern}")
        
        print("\nKey Inclusion Patterns:")
        for pattern in results['patterns'].get('inclusion_patterns', [])[:5]:
            print(f"- {pattern}")
    
    print("\nDetailed results saved to the 'results' directory")