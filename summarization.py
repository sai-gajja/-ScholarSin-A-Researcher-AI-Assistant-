import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import io
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import pyttsx3

# =====================
# ENV & OUTPUT SETTINGS
# =====================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
load_dotenv()

# =====================
# MODEL LOADING - UPDATED TO LaMini-Flan-T5-248M
# =====================
model_path = r"C:\Users\G.SAI\Desktop\ScholarSin\LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Enhanced GPU detection
device = 0 if torch.cuda.is_available() and os.environ.get("USE_GPU", "0") == "1" else -1

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=300,
    min_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    truncation=True,
    repetition_penalty=2.5,
    no_repeat_ngram_size=3
)

llm = HuggingFacePipeline(pipeline=pipe)

# =====================
# IMPROVED PDF TO TEXT FUNCTION
# =====================
def extract_text_from_pdf(pdf_path, chunk_size_value=1000, chunk_overlap_value=100):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_value,
        chunk_overlap=chunk_overlap_value,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    return splitter.split_documents(pages)

# =====================
# ENHANCED SUMMARIZATION LOGIC
# =====================
def summarize_pdf(pdf_path):
    chunks = extract_text_from_pdf(pdf_path)

    def create_summary_prompt(chunk):
        return f"""Summarize the following scientific text about plant disease detection:
        
Text:
{chunk.page_content}

Guidelines:
1. Focus on key methods and findings
2. Maintain technical accuracy
3. Use concise academic language
4. Avoid repetition

Summary:"""

    partial_summaries = []
    for chunk in chunks:
        try:
            result = llm.invoke(create_summary_prompt(chunk))
            if result and isinstance(result, str):
                partial_summaries.append(result.strip())
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            continue

    combined_text = "\n".join(partial_summaries)
    final_prompt = f"""Create a comprehensive research paper summary from these partial summaries:
    
Partial Summaries:
{combined_text}

Instructions:
1. Organize into: Problem, Methods, Results, Implications
2. Maintain flow between sections
3. Keep technical terminology
4. Maximum 400 words

Final Summary:"""

    final_summary = llm.invoke(final_prompt)
    return final_summary if final_summary else "Summary generation failed"

# =====================
# IMPROVED TEXT-TO-SPEECH
# =====================
def text_to_speech(text, save_path=None):
    engine = pyttsx3.init()
    
    # Configure voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 0.9)
    
    # Split long text to avoid buffer issues
    max_chunk = 2000
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i+max_chunk]
        engine.say(chunk)
    
    if save_path:
        engine.save_to_file(text, save_path)
    
    engine.runAndWait()

# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    pdf_path = r"C:\Users\G.SAI\Desktop\ScholarSin_intro.pdf"
    try:
        print("Starting summarization with LaMini-Flan-T5-248M...")
        summary = summarize_pdf(pdf_path)
        
        print("\n=== FINAL SUMMARY ===")
        print(summary)
        
        # Save to file
        with open("research_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        print("\nSummary saved to 'research_summary.txt'")
        
        # Text-to-speech
        print("\nGenerating audio version...")
        text_to_speech(summary, "summary_audio.wav")
        print("Audio saved to 'summary_audio.wav'")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")