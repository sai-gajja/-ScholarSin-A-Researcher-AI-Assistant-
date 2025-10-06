import os
import subprocess
import re
import shutil
import io
import sys
import warnings
import argparse
import torch
from tqdm import tqdm
from textwrap import dedent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

# ===== Configuration =====
DEFAULT_CONFIG = {
    "author": "Sai Siddharth",
    "affiliation": "Department of Computer Science, Example University, India",
    "email": "sai.siddharth@example.com",
    "date": r"\today"
}

# ---------------- LaTeX safety ----------------
def latex_escape(text):
    if not isinstance(text, str):
        return text
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\^{}", "\\": r"\textbackslash{}", "<": r"\textless{}",
        ">": r"\textgreater{}", "|": r"\textbar{}"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# ---------------- PDF handling ----------------
def load_and_chunk_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        return splitter.split_documents(docs)
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")

# ---------------- LLM setup ----------------
def initialize_llm(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        device = 0 if torch.cuda.is_available() else -1
        print(f"Using {'GPU' if device == 0 else 'CPU'} for inference")

        text_generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=1024,
            min_length=200,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3
        )
        return HuggingFacePipeline(pipeline=text_generator)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

# ---------------- Prompts ----------------
def create_prompts():
    return {
        "title": PromptTemplate.from_template(
            "Generate a concise academic title (8-12 words) about: {topic}. Include key technical terms."
        ),
        "abstract": PromptTemplate.from_template(
            "Based on this context:\n{text}\n\nWrite a structured abstract covering: scope, methodology, findings, and gaps. Be concise (~200 words). Avoid citations."
        ),
        "keywords": PromptTemplate.from_template(
            "From this context about {topic}:\n{text}\n\nExtract 5-6 technical keywords, comma-separated."
        ),
        "introduction": PromptTemplate.from_template(
            "From this content:\n{text}\n\nWrite an academic-style introduction to {topic}."
        ),
        "methodology": PromptTemplate.from_template(
            "From this content:\n{text}\n\nSummarize the methodologies used in {topic} research."
        ),
        "results": PromptTemplate.from_template(
            "From this content:\n{text}\n\nSummarize the key results/findings in {topic} research."
        ),
        "discussion": PromptTemplate.from_template(
            "From this content:\n{text}\n\nDiscuss the implications, challenges, and limitations in {topic} research."
        ),
        "conclusion": PromptTemplate.from_template(
            "From this content:\n{text}\n\nWrite a conclusion summarizing main points and future directions for {topic}."
        )
    }

# ---------------- Text processing ----------------
def process_section_content(text, max_chars=2000):
    text = str(text)[:max_chars]
    text = re.sub(r'\s+', ' ', text).strip()
    return latex_escape(text)

# ---------------- Main generation ----------------
def generate_academic_review(pdf_path, research_topic, model_path):
    print("Loading and processing PDF...")
    chunks = load_and_chunk_pdf(pdf_path)
    llm = initialize_llm(model_path)
    prompts = create_prompts()

    context_raw = "\n\n".join([doc.page_content for doc in chunks[:5]])
    context_clean = process_section_content(context_raw, 1800)

    results = {}
    for section, prompt in tqdm(prompts.items(), desc="Generating sections"):
        try:
            chain = prompt | llm
            generated = chain.invoke({"text": context_clean, "topic": research_topic})
            results[section] = process_section_content(generated)
        except Exception as e:
            results[section] = latex_escape(f"[Section generation error: {str(e)}]")
    return results

# ---------------- LaTeX creation ----------------
def create_latex_document(content, config):
    latex_template = r"""
\documentclass[10pt,twocolumn]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{geometry}
\geometry{margin=0.75in}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{hyperref}
\pdfsuppresswarningpagegroup=1

\titleformat{\section}{\bfseries\large}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\bfseries}{\thesubsection}{0.5em}{}

\newcommand{\affiliation}[1]{\begin{center}\normalsize#1\end{center}}

\title{{\bfseries {TITLE}}}
\author{{\normalsize {AUTHOR} \\
\affiliation{{{AFFILIATION}}} \\
\texttt{{{EMAIL}}}}}
\date{{{DATE}}}

\begin{document}
\twocolumn[
\begin{@twocolumnfalse}
\maketitle
\begin{abstract}
{ABSTRACT}
\end{abstract}
\noindent\textbf{Keywords:} {KEYWORDS}
\vspace{1em}
\end{@twocolumnfalse}
]

\section{Introduction}
{INTRODUCTION}

\section{Methodology}
{METHODOLOGY}

\section{Results}
{RESULTS}

\section{Discussion}
{DISCUSSION}

\section{Conclusion}
{CONCLUSION}

\end{document}
"""
    return latex_template.format(
        TITLE=content.get('title', 'Literature Review'),
        AUTHOR=config['author'],
        AFFILIATION=config['affiliation'],
        EMAIL=config['email'],
        DATE=config['date'],
        ABSTRACT=content.get('abstract', ''),
        KEYWORDS=content.get('keywords', ''),
        INTRODUCTION=content.get('introduction', ''),
        METHODOLOGY=content.get('methodology', ''),
        RESULTS=content.get('results', ''),
        DISCUSSION=content.get('discussion', ''),
        CONCLUSION=content.get('conclusion', '')
    )

# ---------------- PDF compilation ----------------
def compile_pdf(tex_file):
    if not shutil.which("pdflatex"):
        raise RuntimeError("pdflatex not found. Install TeX Live or MikTeX.")

    # Auto-update MiKTeX packages to prevent warnings
    if shutil.which("mpm"):
        subprocess.run(["mpm", "--update-db"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["mpm", "--update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Compiling LaTeX document...")
    for _ in range(2):  # Two passes
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(result.stderr)
            return False
    return True

# ---------------- Cleanup ----------------
def cleanup():
    for ext in ['.aux', '.log', '.out']:
        try:
            os.remove(f"review{ext}")
        except FileNotFoundError:
            pass

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Academic Paper Generator")
    parser.add_argument('--pdf', required=True, help='Path to input PDF')
    parser.add_argument('--topic', required=True, help='Research topic')
    parser.add_argument('--model', required=True, help='Path to LLM model')
    parser.add_argument('--output', default='review', help='Output filename (without extension)')
    args = parser.parse_args()

    print("\n=== Academic Paper Generator ===")
    print(f"Input PDF: {args.pdf}")
    print(f"Research Topic: {args.topic}")
    print(f"Model Path: {args.model}\n")

    review_content = generate_academic_review(args.pdf, args.topic, args.model)
    latex_source = create_latex_document(review_content, DEFAULT_CONFIG)

    tex_file = f"{args.output}.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(latex_source)
    print(f"LaTeX source saved to {tex_file}")

    if compile_pdf(tex_file):
        print(f"\nSUCCESS: PDF generated as {args.output}.pdf")
        if sys.platform == "win32":
            os.startfile(f"{args.output}.pdf")
    else:
        print("\nERROR: PDF compilation failed. Check .log file.")

    cleanup()

if __name__ == "__main__":
    main()
