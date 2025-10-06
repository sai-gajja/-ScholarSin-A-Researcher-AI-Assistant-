import os
# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import warnings
import torch
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

def load_and_chunk_pdf(pdf_path):
    """Load and chunk PDF document with improved processing"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    return splitter.split_documents(docs)

def initialize_llm():
    """Initialize LaMini-Flan-T5-248M model for academic writing"""
    model_path = r"C:\Users\G.SAI\Desktop\ScholarSin\LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Configure for academic text generation
    text_generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=512,
        min_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=2.5,  # Increased to reduce repetition
        no_repeat_ngram_size=3,  # Added to prevent n-gram repetition
        num_return_sequences=1
    )
    return HuggingFacePipeline(pipeline=text_generator)

def create_prompts():
    """Structured academic prompts for review paper sections"""
    return {
        "title": PromptTemplate.from_template(
            "Generate a concise, impactful title (10-15 words) for a review paper about: {topic}. "
            "Include key technical terms and scope. Be specific and avoid generic phrases."
        ),
        "abstract": PromptTemplate.from_template(
            """Compose a structured abstract (200-250 words) with these sections:
1. Research Scope: The field of {topic} and its significance
2. Review Methodology: Approach to literature analysis
3. Major Findings: Key insights from reviewed works
4. Research Gaps: Important unanswered questions

Maintain formal academic tone and avoid repetition."""
        ),
        "keywords": PromptTemplate.from_template(
            "Extract 5-8 domain-specific keywords from this text about {topic}. "
            "Format as comma-separated list, ordered by importance. "
            "Include only relevant technical terms."
        ),
        "introduction": PromptTemplate.from_template(
            """Write a comprehensive introduction (500-600 words) covering:
1. Field Overview: Importance of {topic}
2. Historical Context: Evolution of research
3. Current State: Main approaches/technologies
4. Review Objectives: Purpose of this synthesis
5. Paper Organization: Section overview

Reference key literature where appropriate. Avoid redundancy."""
        ),
        "taxonomy": PromptTemplate.from_template(
            """Develop a classification framework for {topic} including:
1. Methodological Categories
2. Theoretical Approaches
3. Application Domains
4. Chronological Development

Present as a structured taxonomy with brief explanations. Be specific."""
        ),
        "critical_analysis": PromptTemplate.from_template(
            """Provide critical analysis of {topic} covering:
1. Comparative Evaluation of Methods
2. Strengths/Limitations of Current Approaches
3. Recurring Challenges
4. Validation Methodologies

Support with evidence from reviewed literature. Be analytical."""
        ),
        "trends": PromptTemplate.from_template(
            """Identify and analyze current trends in {topic}:
1. Emerging Methodologies
2. Technological Advancements
3. Interdisciplinary Connections
4. Shifting Research Focus Areas

Provide concrete examples."""
        ),
        "applications": PromptTemplate.from_template(
            """Discuss practical applications of {topic}:
1. Industry Implementations
2. Case Studies
3. Real-world Impact
4. Commercialization Challenges

Include specific examples where possible."""
        ),
        "future_directions": PromptTemplate.from_template(
            """Propose future research directions for {topic}:
1. Theoretical Frontiers
2. Methodological Innovations
3. Emerging Application Areas
4. Interdisciplinary Opportunities

Prioritize by potential impact. Be specific."""
        ),
        "conclusion": PromptTemplate.from_template(
            """Synthesize a conclusion section for {topic} covering:
1. Key Review Findings
2. Field-wide Implications
3. Recommended Research Agenda
4. Final Reflections

Avoid introducing new concepts. Be concise."""
        )
    }

def post_process(text):
    """Academic writing style enhancements and redundancy removal"""
    improvements = {
        "we found that": "the review reveals that",
        "in this paper": "in this review",
        "our study shows": "the analysis demonstrates",
        "I think": "the evidence suggests",
        "a lot of": "numerous",
        "really good": "highly effective"
    }
    
    # Remove duplicate sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        if sentence not in seen_sentences:
            seen_sentences.add(sentence)
            unique_sentences.append(sentence)
    
    cleaned_text = '. '.join(unique_sentences) + '.' if unique_sentences else text
    
    # Apply style improvements
    for informal, formal in improvements.items():
        cleaned_text = cleaned_text.replace(informal, formal)
    
    return cleaned_text.strip()

def generate_academic_review(pdf_path, research_topic):
    """Generate comprehensive academic literature review"""
    chunks = load_and_chunk_pdf(pdf_path)
    llm = initialize_llm()
    prompts = create_prompts()
    
    results = {}
    context = "\n\n".join([doc.page_content for doc in chunks[:3]])  # Use first few chunks for context
    
    print("\nGenerating review sections:")
    for section, prompt in prompts.items():
        try:
            print(f"- Generating {section}...")
            input_dict = {"text": context, "topic": research_topic}
            chain = prompt | llm
            generated = chain.invoke(input_dict)
            results[section] = post_process(generated)
        except Exception as e:
            print(f"Error generating {section}: {str(e)}")
            results[section] = f"[Content generation failed for {section}]"
    
    # Format as Markdown document
    full_review = f"""
# {results['title']}

**Keywords**: {results['keywords']}

## Abstract
{results['abstract']}

## 1. Introduction
{results['introduction']}

## 2. Research Taxonomy
{results['taxonomy']}

## 3. Critical Analysis
{results['critical_analysis']}

## 4. Current Trends
{results['trends']}

## 5. Practical Applications
{results['applications']}

## 6. Future Directions
{results['future_directions']}

## 7. Conclusion
{results['conclusion']}
"""
    return full_review

if __name__ == "__main__":
    pdf_path = r"C:\Users\G.SAI\Desktop\trimmed_scholarsin.pdf"
    research_topic = "Review on Neural Signal Decoding"
    
    try:
        print("Initializing LaMini-Flan-T5-248M model...")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        print("\nStarting academic literature review generation...")
        review = generate_academic_review(pdf_path, research_topic)
        
        with open("academic_review.md", "w", encoding="utf-8") as f:
            f.write(review)
        
        print("\n=== ACADEMIC REVIEW GENERATED SUCCESSFULLY ===")
        print("Saved to 'academic_review.md'")
        
        # Print the complete review
        print("\n=== COMPLETE REVIEW ===\n")
        print(review)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Review generation failed. Please check inputs and try again.")