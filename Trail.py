import os
import subprocess
from textwrap import dedent

# ===== Metadata =====
TITLE  = "Review of ML and DL in Disease Detection"
AUTHOR = "Sai Siddharth"
AFFILIATION = "Department of Computer Science, Example University, India"
EMAIL = "sai.siddharth@example.com"
DATE   = r"\today"

# ===== Section content placeholders =====
ABSTRACT = dedent("""
This review synthesizes recent progress at the intersection of machine learning and deep learning
for disease detection. It outlines methodological foundations, surveys key application domains,
critiques validation practices, and highlights open challenges and future directions.
""").strip()

KEYWORDS = "machine learning, deep learning, disease detection, medical imaging, diagnostics"

INTRODUCTION = """
Provide the field overview, historical context, current state, review objectives, and paper organization.
"""

TAXONOMY = """
Methodological categories, theoretical approaches, application domains, and chronological development.
"""

CRITICAL_ANALYSIS = """
Comparative evaluation, strengths/limitations, recurring challenges, validation methodologies.
"""

TRENDS = """
Emerging methods, technological advances, interdisciplinary links, shifting focus areas.
"""

APPLICATIONS = """
Industry implementations, case studies, real-world impact, commercialization challenges.
"""

FUTURE_DIRECTIONS = """
Theoretical frontiers, methodological innovations, emerging application areas, interdisciplinary opportunities.
"""

CONCLUSION = """
Synthesize key findings, implications, research agenda, and final reflections.
"""

# ===== LaTeX Template =====
latex_template = r"""
\documentclass[10pt,twocolumn]{{article}}

% ---- Packages ----
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{times}} % Times New Roman-like font
\usepackage{{geometry}}
\geometry{{margin=0.75in}}
\usepackage{{titlesec}}
\usepackage{{enumitem}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{lipsum}}

% ---- Section formatting ----
\titleformat{{\section}}{{\bfseries\large}}{{\thesection.}}{{0.5em}}{{}}
\titleformat{{\subsection}}{{\bfseries}}{{\thesubsection}}{{0.5em}}{{}}

% ---- Title block ----
\newcommand{{\affiliation}}[1]{{\begin{{center}}\normalsize#1\end{{center}}}}

\title{{\bfseries {TITLE}}}
\author{{\normalsize {AUTHOR} \\
\affiliation{{{AFFILIATION}}} \\
\texttt{{{EMAIL}}}}}
\date{{{DATE}}}

\begin{{document}}
\twocolumn[
\maketitle
\begin{{onecolumn}}
\begin{{abstract}}
{ABSTRACT}
\end{{abstract}}
\noindent\textbf{{Keywords:}} {KEYWORDS}
\vspace{{1em}}
\end{{onecolumn}}
\twocolumn
]

\section{{Introduction}}
{INTRODUCTION}

\section{{Research Taxonomy}}
{TAXONOMY}

\section{{Critical Analysis}}
{CRITICAL_ANALYSIS}

\section{{Current Trends}}
{TRENDS}

\section{{Practical Applications}}
{APPLICATIONS}

\section{{Future Directions}}
{FUTURE_DIRECTIONS}

\section{{Conclusion}}
{CONCLUSION}

\end{{document}}
"""

# ===== Format the template =====
latex_source = latex_template.format(
    TITLE=TITLE,
    AUTHOR=AUTHOR,
    AFFILIATION=AFFILIATION,
    EMAIL=EMAIL,
    DATE=DATE,
    ABSTRACT=ABSTRACT,
    KEYWORDS=KEYWORDS,
    INTRODUCTION=INTRODUCTION,
    TAXONOMY=TAXONOMY,
    CRITICAL_ANALYSIS=CRITICAL_ANALYSIS,
    TRENDS=TRENDS,
    APPLICATIONS=APPLICATIONS,
    FUTURE_DIRECTIONS=FUTURE_DIRECTIONS,
    CONCLUSION=CONCLUSION
)

# ===== Save LaTeX file =====
tex_filename = "review_paper.tex"
pdf_filename = "review_paper.pdf"

with open(tex_filename, "w", encoding="utf-8") as f:
    f.write(latex_source)

print(f"[OK] Wrote {tex_filename}")

# ===== Compile with pdflatex =====
def compile_pdf(tex_file: str) -> bool:
    # ✅ Full path to MiKTeX pdflatex
    pdflatex_path = r"C:\Users\G.SAI\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    
    try:
        for i in range(2):
            print(f"[pdflatex] Pass {i+1} ...")
            proc = subprocess.run(
                [pdflatex_path, "-interaction=nonstopmode", tex_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            tail = "\n".join(proc.stdout.splitlines()[-10:])
            print(tail)
            if proc.returncode != 0:
                return False
        return True
    except FileNotFoundError:
        print("[ERROR] pdflatex not found at:", pdflatex_path)
        return False

if compile_pdf(tex_filename):
    if os.path.exists(pdf_filename):
        print(f"[OK] PDF generated: {os.path.abspath(pdf_filename)}")
import os
import subprocess
from textwrap import dedent

# ===== Metadata =====
TITLE = "Review of ML and DL in Disease Detection"
AUTHOR = "Sai Siddharth"
AFFILIATION = "Department of Computer Science, Example University, India"
EMAIL = "sai.siddharth@example.com"
DATE = r"\today"

# ===== Section content placeholders =====
ABSTRACT = dedent("""
This review synthesizes recent progress at the intersection of machine learning and deep learning
for disease detection. It outlines methodological foundations, surveys key application domains,
critiques validation practices, and highlights open challenges and future directions.
""").strip()

KEYWORDS = "machine learning, deep learning, disease detection, medical imaging, diagnostics"

INTRODUCTION = """
Provide the field overview, historical context, current state, review objectives, and paper organization.
"""

TAXONOMY = """
Methodological categories, theoretical approaches, application domains, and chronological development.
"""

CRITICAL_ANALYSIS = """
Comparative evaluation, strengths/limitations, recurring challenges, validation methodologies.
"""

TRENDS = """
Emerging methods, technological advances, interdisciplinary links, shifting focus areas.
"""

APPLICATIONS = """
Industry implementations, case studies, real-world impact, commercialization challenges.
"""

FUTURE_DIRECTIONS = """
Theoretical frontiers, methodological innovations, emerging application areas, interdisciplinary opportunities.
"""

CONCLUSION = """
Synthesize key findings, implications, research agenda, and final reflections.
"""

# ===== Simplified LaTeX Template =====
latex_template = r"""
\documentclass[10pt,twocolumn]{article}

% ---- Basic Packages ----
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{times} % Times New Roman-like font
\usepackage{geometry}
\geometry{margin=0.75in}
\usepackage{graphicx}
\usepackage{lipsum} % For placeholder text

% ---- Section formatting ----
\usepackage{titlesec}
\titleformat{\section}{\bfseries\large}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\bfseries}{\thesubsection}{0.5em}{}

% ---- Title block ----
\newcommand{\affiliation}[1]{\begin{center}\normalsize#1\end{center}}

\title{\bfseries {TITLE}}
\author{\normalsize {AUTHOR} \\
\affiliation{{AFFILIATION}} \\
\texttt{{EMAIL}}}
\date{{DATE}}

\begin{document}
\twocolumn[
\maketitle
\begin{onecolumn}
\begin{abstract}
{ABSTRACT}
\end{abstract}
\noindent\textbf{Keywords:} {KEYWORDS}
\vspace{1em}
\end{onecolumn}
\twocolumn
]

\section{Introduction}
{INTRODUCTION}

\section{Research Taxonomy}
{TAXONOMY}

\section{Critical Analysis}
{CRITICAL_ANALYSIS}

\section{Current Trends}
{TRENDS}

\section{Practical Applications}
{APPLICATIONS}

\section{Future Directions}
{FUTURE_DIRECTIONS}

\section{Conclusion}
{CONCLUSION}

\end{document}
"""

# ===== Format the template =====
latex_source = latex_template.format(
    TITLE=TITLE,
    AUTHOR=AUTHOR,
    AFFILIATION=AFFILIATION,
    EMAIL=EMAIL,
    DATE=DATE,
    ABSTRACT=ABSTRACT,
    KEYWORDS=KEYWORDS,
    INTRODUCTION=INTRODUCTION,
    TAXONOMY=TAXONOMY,
    CRITICAL_ANALYSIS=CRITICAL_ANALYSIS,
    TRENDS=TRENDS,
    APPLICATIONS=APPLICATIONS,
    FUTURE_DIRECTIONS=FUTURE_DIRECTIONS,
    CONCLUSION=CONCLUSION
)

# ===== Save LaTeX file =====
tex_filename = "review_paper.tex"
pdf_filename = "review_paper.pdf"

with open(tex_filename, "w", encoding="utf-8") as f:
    f.write(latex_source)

print(f"[OK] Wrote {tex_filename}")

# ===== Compile with pdflatex (using system PATH) =====
def compile_pdf(tex_file: str) -> bool:
    try:
        for i in range(2):  # Run twice to resolve references
            print(f"[pdflatex] Pass {i+1} ...")
            proc = subprocess.run(
    ["pdflatex", "--enable-installer", "-interaction=nonstopmode", tex_filename],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)
            if proc.returncode != 0:
                print(f"[ERROR] pdflatex failed with return code {proc.returncode}")
                print("Last 10 lines of output:")
                print("\n".join(proc.stdout.splitlines()[-10:]))
                return False
        return True
    except FileNotFoundError:
        print("[ERROR] pdflatex not found in PATH. Please ensure MiKTeX is installed and in your system PATH.")
        return False

if compile_pdf(tex_filename):
    if os.path.exists(pdf_filename):
        print(f"[OK] PDF generated: {os.path.abspath(pdf_filename)}")
    else:
        print("[ERROR] pdflatex ran but no PDF was produced")
else:
    print("[ERROR] Failed to compile PDF")