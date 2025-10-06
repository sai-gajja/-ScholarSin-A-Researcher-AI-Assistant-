import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import io
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from tqdm import tqdm
from collections import Counter
import torch

# =====================
# ENV & OUTPUT SETTINGS
# =====================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
load_dotenv()

# =====================
# MODEL LOADING (LaMini-Flan-T5-248M)
# =====================
model_path = r"C:\Users\G.SAI\Desktop\ScholarSin\LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Device configuration
device = 0 if torch.cuda.is_available() else -1

# QA Pipeline configuration (using text2text-generation)
qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=300,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    repetition_penalty=2.5
)

# =====================
# PDF PROCESSING (Optimized for Generative QA)
# =====================
def process_pdf(pdf_path):
    """Load and chunk PDF with settings optimized for generative QA"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = ' '.join([page.page_content.replace('\n', ' ') for page in pages])
    
    # Larger chunks work better for generative QA
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", ". ", "! ", "? ", "\n", " ", ""],
        length_function=len
    )
    return splitter.split_text(full_text)

# =====================
# QUESTION ANSWERING (Generative Approach)
# =====================
def answer_from_pdf(question, chunks):
    """Generate answers using the T5 model"""
    answers = []
    for chunk in tqdm(chunks, desc="Analyzing document"):
        try:
            # Format the input for generative QA
            input_text = f"question: {question} context: {chunk}"
            result = qa_pipeline(input_text)[0]['generated_text']
            answers.append(result)
        except Exception as e:
            continue
    
    if not answers:
        return "The answer could not be generated from the document."
    
    # Return the most common answer (simple consensus)
    answer_counts = Counter(answers)
    best_answer = answer_counts.most_common(1)[0][0]
    return best_answer

# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    pdf_path = r"C:\Users\G.SAI\Desktop\Plant_Disease_Detection_Report_elevate_labs_Final_Project.pdf"
    chunks = process_pdf(pdf_path)
    
    print("Document loaded. Ask questions (type 'quit' to exit)")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ('quit', 'exit'):
            break
            
        answer = answer_from_pdf(question, chunks)
        print(f"\nAnswer: {answer}\n{'='*50}")