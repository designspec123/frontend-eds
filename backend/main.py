import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSemanticPreservingSplitter
from langchain.schema import Document
from bs4 import BeautifulSoup, Tag
import requests
from enum import Enum

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere

st.set_page_config(layout="wide")

#from fetch_dependent_classes import fetch_dependent_classes,get_css_classes_used_from_chunk

# Define directories
INPUT_DIR = "input_files"
DESIGN_SPEC_DIR = "design_spec"
OUTPUT_DIR = "output_files"
FAISS_INDEX_PATH_CSS = "faiss_index_css"
FAISS_INDEX_PATH_YAML = "faiss_index_yaml"
FAISS_INDEX_PATH = "faiss_index"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize LLM and vector EmBeddings Begin***********************************
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
ds_components = ["button", "card", "container", "heading", "image", "list", "navbar", "timeline" , "script" ,"text" , "sidebar"]
class LLMProvider(Enum):
    GROQ = "gemma2-9b-it"#"llama3-70b-8192" #"llama3-8b-instruct" #"gemma2-9b-it" "deepseek-r1-distill-llama-70b"
    GEMINI = "gemma-3-27b-it" #gemma-3-27b-it gemini-2.0-flash-lite 
    OPENAI = "gpt-4o-mini"  # Or "gpt-4" or "gpt-3.5-turbo"
    COHERE = "command-a-03-2025"

load_dotenv()
SELECTED_LLM = LLMProvider.GEMINI  # Change to LLMProvider.GROQ to use Groq

if SELECTED_LLM == LLMProvider.GROQ:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=SELECTED_LLM.value,
        temperature=0.1
    )

elif SELECTED_LLM == LLMProvider.GEMINI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
    llm = ChatGoogleGenerativeAI(
        model=SELECTED_LLM.value,
        google_api_key=api_key,
        temperature=0.1
    )
elif SELECTED_LLM == LLMProvider.OPENAI:
    api_key = os.getenv("OPENAI_API_KEY")
    print("openai")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found.")
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=SELECTED_LLM.value,
        temperature=0.1
    )
elif SELECTED_LLM == LLMProvider.COHERE:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found.")
    llm = ChatCohere(
        cohere_api_key=api_key,
        model_name=SELECTED_LLM.value,
        temperature=0.1
    )

else:
    raise ValueError("Invalid LLM provider selected.")


# Initialize LLM and vector EmBeddings End*************************************************

# Utility Functions Begin *****************************************************************

def read_files(directory, allowed_extensions=None):
  
    if allowed_extensions is None:
        allowed_extensions = {".yaml", ".css"}

    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in allowed_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        component_name = os.path.splitext(file)[0]  # removes ".css" or ".yaml"
                        print("component_name metadada: " + component_name)
                        docs.append(Document(
                            page_content=content,
                            metadata={"component_name": component_name}
                        ))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return docs



def load_or_create_faiss_index(docs, faiss_index_path):
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(faiss_index_path)
    return vectorstore


chunk_counter=0

def retrieve_design_standard(query: str, vectorstore) -> str:
    """
    Retrieves documents whose metadata['filename'] matches any comma-separated term in the query.
    """
    terms = [term.strip().lower() for term in query.split(",")]
    results = []

    for term in terms:
        docs = vectorstore.similarity_search(term, k=20)  # broad match
        matched = []
        for doc in docs:
            component_name = doc.metadata.get("component_name", "").lower()
            #print("component_name termname "+ component_name + term)  # Debugging output
            if component_name == term:
                #print("component_name matched termname "+ component_name + term)  # Debugging output
                matched.append(doc)

        if matched:
            section = f"--- Results for component '{term}' ---\n"
            section += "\n\n".join(doc.page_content for doc in matched)
            results.append(section)

    return "\n\n".join(results) if results else ("No relevant component matches found: " + query)



def load_custom_design_spec_css():
    custom_design_spec_css_path = os.path.join(DESIGN_SPEC_DIR, "poc-ds.css")
    if os.path.exists(custom_design_spec_css_path):
        with open(custom_design_spec_css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_custom_design_spec_js():
    custom_design_spec_js_path = os.path.join(DESIGN_SPEC_DIR, "custom_design_spec.bundle.min.js")
    if os.path.exists(custom_design_spec_js_path):
        with open(custom_design_spec_js_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_output_requirements(filename):
    instruction_file = os.path.join(DESIGN_SPEC_DIR, f"{os.path.splitext(filename)[0]}.txt")
    if os.path.exists(instruction_file):
        with open(instruction_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return "No specific Output Requirements found. Proceed with standard transformation."

def fetch_html_as_document(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Document(page_content=response.text, metadata={"filename": url})
    except Exception as e:
        st.error(f"Failed to fetch HTML: {e}")
        return None

def extract_body_only(doc: Document):
    soup = BeautifulSoup(doc.page_content, "html.parser")
    body = soup.body
    if body:
        return Document(page_content=str(body), metadata=doc.metadata)
    else:
        st.warning("No <body> found in the page.")
        return None

def save_chunks_to_file_old(output_folder_path, chunks, filename="chunks_info.txt"):
    output_path = os.path.join(output_folder_path, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n\n--- Chunk {i+1} ---\n")
            f.write(str(chunk))
            
def save_chunks_to_file(output_folder_path, chunks, filename="chunks_info.txt"):
    output_path = os.path.join(output_folder_path, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Write the selected LLM info at the top
        f.write(f"Selected LLM: {SELECTED_LLM.name} ({SELECTED_LLM.value})\n")
        f.write("=" * 50 + "\n\n")

        # Write the chunks
        for i, chunk in enumerate(chunks):
            f.write(f"\n\n--- Chunk {i+1} ---\n")
            f.write(str(chunk))

            
def create_output_folder_path(output_folder_name):
     output_folder_path = os.path.join(OUTPUT_DIR, output_folder_name)
     if not os.path.exists(output_folder_path):
         try:
            os.makedirs(output_folder_path)
            print(f"Created folder: {output_folder_path}")
         except Exception as e:
            print(f"Error creating folder {output_folder_path}: {e}")
            return
     return output_folder_path

def strip_outer_tags(html_block):
    html_block = re.sub(r"<\/?(html|head|body)[^>]*>", "", html_block, flags=re.IGNORECASE)
    return html_block.strip()


def save_correction_prompt_response_to_file(output_folder_path, correction_chunk_filename, correction_prompt, correction_prompt_response, corrected_output_requirements_str):
    
    correction_prompt_response_path = os.path.join(output_folder_path, correction_chunk_filename + "_correction_prompt_reponse.txt")
    with open(correction_prompt_response_path, "w", encoding="utf-8") as f:
        f.write(correction_prompt_response)

    chunk_file_path = os.path.join(output_folder_path, correction_chunk_filename + "_correction_prompt.txt")
    with open(chunk_file_path, "w", encoding="utf-8") as f:
        for item in correction_prompt:
            role = item["role"]
            content = item["content"]

            if role == "user":
                # Add extra newline before each '###', preserving escaped \n
                content_lines = content.split("\\n")
                updated_lines = []
                for line in content_lines:
                    if line.strip().startswith("###"):
                        updated_lines.append("")  # blank line before header
                    updated_lines.append(line)
                # Join back with \n to keep content escaped
                content = "\\n".join(updated_lines)

            # Output in exact dict-like string format
            f.write(f"{{'role': '{role}', 'content': '{content}'}}\n\n")

def save_prompt_response_to_file(output_folder_path, chunk_filename, formatted_prompt, response, summary):
    chunk_file_path = os.path.join(output_folder_path, chunk_filename + "_.txt")
    chunk_reponse_file_path = os.path.join(output_folder_path, chunk_filename + "_response.htm")
    with open(chunk_reponse_file_path, "w", encoding="utf-8") as f:
        f.write(response)

    with open(chunk_file_path, "w", encoding="utf-8") as f:
        for item in formatted_prompt:
            role = item["role"]
            content = item["content"]

            if role == "user":
                # Add extra newline before each '###', preserving escaped \n
                content_lines = content.split("\\n")
                updated_lines = []
                for line in content_lines:
                    if line.strip().startswith("###"):
                        updated_lines.append("")  # blank line before header
                    updated_lines.append(line)
                # Join back with \n to keep content escaped
                content = "\\n".join(updated_lines)

            # Output in exact dict-like string format
            f.write(f"{{'role': '{role}', 'content': '{content}'}}\n\n")
        
        #f.write(f"# Response:\n{response}\n")

def inline_css_from_file(html_content, css_file_path):
    """
    Replaces <link rel="stylesheet" href="poc-ds.css"> with minified inline CSS wrapped in <style> tags.

    Args:
        html_content (str): The original HTML content.
        css_file_path (str): Path to the CSS file to inline.

    Returns:
        str: HTML content with inline CSS.
    """
    import os
    import re

    # Check if file exists
    if not os.path.exists(css_file_path):
        raise FileNotFoundError(f"CSS file not found: {css_file_path}")

    # Read CSS file contents
    with open(css_file_path, "r", encoding="utf-8") as f:
        css_content = f.read()

    # Minify CSS: remove comments, collapse whitespace, tighten spaces
    css_minified = re.sub(r'/\*.*?\*/', '', css_content, flags=re.DOTALL)  # Remove /* comments */
    css_minified = re.sub(r'\s+', ' ', css_minified)                      # Collapse all whitespace
    css_minified = re.sub(r'\s*([{};:,])\s*', r'\1', css_minified)         # Tighten spaces around {}:;,
    css_minified = css_minified.strip()

    # Wrap minified CSS in <style> tag
    inline_style_tag = f"<style>{css_minified}</style>"

    # Replace <link rel="stylesheet" href="poc-ds.css"> in HTML
    html_content_inlined = re.sub(
        r'<link\s+rel=["\']stylesheet["\']\s+href=["\']poc-ds\.css["\']\s*>',
        inline_style_tag,
        html_content,
        flags=re.IGNORECASE
    )

    return html_content_inlined



    
# Utility Functions End ***********************************************************************************************************************************

# Prompt Begin ***********************************************************************************************************************************

def create_join_prompt(transformed_chunks, filename=""):
    html_sections = "\n".join(transformed_chunks)
    join_prompt = f"""
You are a UI transformation assistant.

### Instructions:
- Combine the provided HTML sections into a single coherent <body> structure.
- Ensure proper layout, spacing, responsiveness, and component alignment.
- Do not include <html>, <head>, <link>, or <script> tags. Only return the final <body> content.

### Provided HTML Sections:
{html_sections}
""".strip()

    safe_filename = filename.replace(" ", "_").replace("/", "_")
    log_file_path = os.path.join(OUTPUT_DIR, f"join_prompt_log_{safe_filename}.txt")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(join_prompt)

    return join_prompt


def create_alignment_issue_prompt(original_body, transformed_body, filename="", output_dir="output"):
    """
    Generate a prompt to compare original and transformed HTML bodies for alignment issues,
    and save the prompt to a specified folder.

    Parameters:
    - original_body (str): The original HTML body content.
    - transformed_body (str): The transformed HTML body content to compare.
    - filename (str): Optional base filename for the log file.
    - output_dir (str): Directory to save the output prompt log.

    Returns:
    - str: The generated prompt.
    """
    prompt = f"""
You are a frontend code reviewer specializing in UI/UX alignment.

### Task:
Compare the **original HTML body layout** with the **transformed HTML body** and **list all alignment-related issues** introduced in the transformation.

### Focus Areas:
- Sidebar and main content alignment (e.g., overlaps, spacing, fixed width)
- Responsive layout behavior on small vs. large screens
- Padding, margin, and spacing inconsistencies
- Navbar placement and stacking behavior
- Any structural misalignment affecting readability or layout consistency

### Constraints:
- Do not rewrite the HTML.
- Only list the **differences that affect visual alignment or layout**.
- Keep your response concise and structured as a bullet list.

---

### Original HTML Body:
{original_body}

---

### Transformed HTML Body:
{transformed_body}
""".strip()

    safe_filename = filename.replace(" ", "_").replace("/", "_") or "alignment_issues"
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, f"alignment_issue_prompt_log_{safe_filename}.txt")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    return prompt



def create_unified_prompt(design_standard, ui_instructions, input_content, dependencies, summary):
    return f"""You are a UI transformation assistant.
Generate HTML strictly following the provided design standard.

### Rules:
- Use only the classes and styles defined in the design standard (no external styles).
- Provide only the <body> content (omit <html>, <head> unless specified).
- Keep output minimal while maintaining the correct structure.
- Ensure responsiveness and custom_design_spec compliance.
- Always wrap such blocks with <header>, <section>, or <div> using appropriate custom_design_spec spacing utilities.
- Consider Input HTML Chunk Description for transformation of User Request.
- Consider Retrieved Context Reference if applicable for input HTML chunk.
- Do not miss out on any elements from input chunk during transformation

### Additional Rules:
{ui_instructions}

### Retrieved Context Reference For Transformation:
{design_standard}

### Input HTML Chunk Description:
{summary}

### Input HTML Chunk:
{input_content}

### Output:
""".strip()

def create_correction_prompt(transformation_prompt, generated_html_output):
    if isinstance(transformation_prompt, list):
        user_content = next((item["content"] for item in transformation_prompt if item["role"] == "user"), "")
    else:
        user_content = transformation_prompt

    first_requirement = "Retain original visual and layout structure as described."
    

    system_prompt = """
You are a UI transformation advisor.
Your job is to review the Transformed HTML Output generated from a transformation prompt.
.

Your task is to:
- Identify any discrepancies between the intended design and the HTML output.
- If any issues are found, return them clearly labeled as bullet points under the heading: `# Complaint Issue List`
- If no issues are found, return only the heading: `# No Complaint Issue`
- Then, in all cases, append a section `## MANDATORY OUTPUT RULES:` with:
  - Do not miss and keep all the MANDATORY OUTPUT RULES from the original prompt
  - Additional improvements based on your analysis

Your final response should look like this:

# Complaint Issue List
* **Issue Name:** Explanation of the issue.

## MANDATORY OUTPUT RULES:
- First requirement from original prompt
- Fix for issue 1
- Fix for issue 2
...
""".strip()

    user_prompt = f"""
### Original Transformation Prompt:
{transformation_prompt}

### Transformed HTML Output:
{generated_html_output}

### Instructions:
- Return a markdown report starting with either `# Complaint Issue List` or `# No Complaint Issue`.
- List each issue clearly with a short label and detailed explanation.
- Then, write the `## MANDATORY OUTPUT RULES:` section:
  - Start with the first bullet point from the original prompt (preserved as-is).
  - Then add new bullet points derived from the issues you've listed.
- Do not include any other sections or explanations.
""".strip()

    return {
        "system": system_prompt,
        "user": user_prompt,
        "first_requirement": first_requirement
    }

def create_transform_prompt_old(output_requirements, input_content, summary, css_reference, yaml_reference):
    system_prompt = (
        "You are a UI/UX transformation expert who is capable of transforming HTML pages from one format to another.\n\n"
        "You will be provided with 5 sections:\n"
        "- The input HTML chunk\n"
        "- Input HTML Chunk DescriptionL\n"
        "- CSS for output html\n"
        "- Optional Output html format\n"
        "- MANDATORY OUTPUT RULES\n\n"
        "Your task is to transform the input HTML chunk into compliant output HTML format <body> content only based on the above sections"
    )

    user_prompt = f"""
Your task is to transform the below input HTML chunk into compliant output HTML `<body>` content only.  
**Do not include JSON, explanations, or anything else — only return the transformed HTML.**

### Input HTML Chunk:
{input_content}

### Input HTML Chunk Description:
{summary}

### CSS for output html:
{css_reference}

### Output html format :
{yaml_reference}

### MANDATORY OUTPUT RULES:
{output_requirements}

### Output:
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def create_transform_prompt(output_requirements, input_content, summary, css_reference, yaml_reference):
    user_prompt = f"""
You are a UI/UX transformation expert capable of transforming HTML pages from one format to another.

You will be provided with 5 sections:
- The input HTML chunk
- Input HTML Chunk Description
- CSS for output HTML
- Optional Output HTML format
- MANDATORY OUTPUT RULES

Your task is to transform the input HTML chunk into compliant output HTML `<body>` content only, based on the above sections.

---

### Rules:
- Do not include <html>, <head>, <link>, or <script> tags. Only return the <body> content.
- Use only the CSS classes and styles defined in the provided CSS reference.
- Follow the optional output format if provided.
- Strictly adhere to all MANDATORY OUTPUT RULES.
- Keep the output minimal but semantically correct and fully responsive.

---

### Input HTML Chunk:
{input_content}

---

### Input HTML Chunk Description:
{summary}

---

### CSS for Output HTML:
{css_reference}

---

### Optional Output HTML Format:
{yaml_reference}

---

### MANDATORY OUTPUT RULES:
{output_requirements}

---

### Output:
""".strip()

    return [{"role": "user", "content": user_prompt}]


def summarize_html_chunk_old(html_chunk: str) -> str:
    summarization_prompt = f"""
You are a UI layout interpreter. Your task is to read raw HTML and summarize what kind of component it represents using simple, developer-friendly terms.

Summarize the following HTML in one concise but detailed sentence.
Include key class names, inline styles, layout and alignment behaviors, colors, interactions, and visible text content (like button labels or headings).
The summary should be short but informative enough for later HTML or design spec reconstruction.

### Example:
HTML:
<header class="w3-container w3-red w3-center" style="padding:128px 16px">
<h1 class="w3-margin w3-jumbo">START PAGE</h1>
<p class="w3-xlarge">Template by w3.css</p>
<button class="w3-button w3-black w3-padding-large w3-large w3-margin-top">Get Started</button>
</header>

Summary:
Centered red header (w3-container w3-red w3-center, padding:128px 16px) with title "START PAGE" (w3-jumbo), subtext "Template by w3.css" (w3-xlarge), and a large black "Get Started" button (w3-button w3-black w3-padding-large).

### Now summarize the following HTML:

{html_chunk}

Summary:
""".strip()

    try:
        response = llm.invoke(summarization_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Summarization failed: {e}")
        return ""

def summarize_html_chunk_old(html_chunk: str, allowed_components: list[str]) -> tuple[str, str]:
    allowed_components_str = ", ".join(allowed_components)

    system_prompt = (
        "You are a UI layout interpreter. Your task is to read raw HTML and summarize what kind of component it represents using simple, developer-friendly terms.\n\n"
        "Summarize the HTML in one concise but detailed sentence. Include:\n"
        "- Key class names\n"
        "- Inline styles\n"
        "- Layout and alignment behaviors\n"
        "- Colors\n"
        "- Interactions\n"
        "- Visible text content (like button labels or headings)\n\n"
        "The summary should be short but informative enough for a developer to reconstruct the HTML or design spec later.\n\n"
        "Use this format in your response:\n"
        "Summary: <concise description here>\n"
        "Components: <comma-separated components>\n\n"
        f"Only choose components from this list:\n{allowed_components_str}\n\n"
        "If the HTML includes tags or patterns like <nav>, <img>, <ul>, or terms like 'navigation bar', 'photo', etc., intelligently map them to the most appropriate component from the list.\n"
        "Do not use raw tag names like 'div', 'header', or 'section' — only map to valid components."
    )

    user_prompt = f"""
### Example:
HTML:
<header class="w3-container w3-red w3-center" style="padding:128px 16px">
  <h1 class="w3-margin w3-jumbo">START PAGE</h1>
  <p class="w3-xlarge">Template by w3.css</p>
  <button class="w3-button w3-black w3-padding-large w3-large w3-margin-top">Get Started</button>
</header>

Summary: Centered red header (w3-container w3-red w3-center, padding:128px 16px) with title "START PAGE" (w3-jumbo), subtext "Template by w3.css" (w3-xlarge), and a large black 'Get Started' button (w3-button w3-black w3-padding-large).
Components: container, heading, button

### Now summarize the following HTML:
{html_chunk}
""".strip()

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        content = response.content.strip()

        summary = ""
        components = ""

        for line in content.splitlines():
            if line.lower().startswith("summary:"):
                summary = line[len("summary:"):].strip()
            elif line.lower().startswith("components:"):
                components = line[len("components:"):].strip()

        return summary, components

    except Exception as e:
        print(f"Summarization failed: {e}")
        return "", ""

def summarize_html_chunk(html_chunk: str, allowed_components: list[str]) -> tuple[str, str]:
    allowed_components_str = ", ".join(allowed_components)

    user_prompt = f"""
You are a UI layout interpreter. Your task is to read raw HTML and summarize what kind of component it represents using simple, developer-friendly terms.

Summarize the HTML in one concise but detailed sentence. Include:
- Key class names
- Inline styles
- Layout and alignment behaviors
- Colors
- Interactions
- Visible text content (like button labels or headings)

The summary should be short but informative enough for a developer to reconstruct the HTML or design spec later.

Use this format in your response:
Summary: <concise description here>
Components: <comma-separated components>

Only choose components from this list:
{allowed_components_str}

If the HTML includes tags or patterns like <nav>, <img>, <ul>, or terms like 'navigation bar', 'photo', etc., intelligently map them to the most appropriate component from the list.
Do not use raw tag names like 'div', 'header', or 'section' — only map to valid components.

---

### Example:
HTML:
<header class="w3-container w3-red w3-center" style="padding:128px 16px">
  <h1 class="w3-margin w3-jumbo">START PAGE</h1>
  <p class="w3-xlarge">Template by w3.css</p>
  <button class="w3-button w3-black w3-padding-large w3-large w3-margin-top">Get Started</button>
</header>

Summary: Centered red header (w3-container w3-red w3-center, padding:128px 16px) with title "START PAGE" (w3-jumbo), subtext "Template by w3.css" (w3-xlarge), and a large black 'Get Started' button (w3-button w3-black w3-padding-large).
Components: container, heading, button

---

### Now summarize the following HTML:
{html_chunk}
""".strip()

    try:
        response = llm.invoke(user_prompt)
        content = response.content.strip()

        summary = ""
        components = ""

        for line in content.splitlines():
            if line.lower().startswith("summary:"):
                summary = line[len("summary:"):].strip()
            elif line.lower().startswith("components:"):
                components = line[len("components:"):].strip()

        return summary, components

    except Exception as e:
        print(f"Summarization failed: {e}")
        return "", ""





#Promt END *****************************************************************************************************************

docs_with_css_yaml = ""
docs_with_yaml = ""
docs_with_css= ""
if not os.path.exists(FAISS_INDEX_PATH):
    print("read_files(DESIGN_SPEC_DIR)")
    docs_with_css_yaml = read_files(DESIGN_SPEC_DIR)

if not os.path.exists(FAISS_INDEX_PATH_YAML):
    print("read_files(DESIGN_SPEC_DIR yaml)")
    docs_with_yaml = read_files(DESIGN_SPEC_DIR, allowed_extensions={".yaml"})

if not os.path.exists(FAISS_INDEX_PATH_CSS):
    print("read_files(DESIGN_SPEC_DIR css)")
    docs_with_css = read_files(DESIGN_SPEC_DIR, allowed_extensions={".css"})

vectorstore = load_or_create_faiss_index(docs_with_css_yaml,FAISS_INDEX_PATH)
vectorstore_with_yaml = load_or_create_faiss_index(docs_with_yaml,FAISS_INDEX_PATH_YAML)
vectorstore_with_css = load_or_create_faiss_index(docs_with_css,FAISS_INDEX_PATH_CSS)

def generate_components(generate_component_command):
    retrieved_docs = vectorstore.similarity_search(generate_component_command, k=2)
    design_standard_retrieved = "\n".join([
        "\n".join([line for line in doc.page_content.split("\n") if "." in line][:5])
        for doc in retrieved_docs
    ])

    ui_specific_instructions = "Component should be simple, structured, and reusable."
    formatted_prompt = create_unified_prompt(design_standard_retrieved,ui_specific_instructions,generate_component_command,"-NONE-")

    response = ""
    for chunk in llm.stream(formatted_prompt):
        response += chunk.content

    cleaned_response = response.replace("```html", "").replace("```", "").strip()
    custom_design_spec_css = load_custom_design_spec_css()

    final_html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
       <style>
            {custom_design_spec_css}
        </style>
        <title>Generated UI Component</title>
    </head>
    <body>
        {cleaned_response}
    </body>
    </html>
    """

    output_file = os.path.join(OUTPUT_DIR, f"generate_{len(os.listdir(OUTPUT_DIR))}.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)

    st.subheader("Generated custom_design_spec Output")
    st.session_state.custom_design_spec_code = final_html
    st.rerun()
#******************************************************************************************************************************


def extract_html_body(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.body
    return body




def extract_body_chunks(html_content, max_chunk_size=500):
    #output_file_path = os.path.join(OUTPUT_DIR, f"html_str.txt")
    #with open(output_file_path, "w", encoding="utf-8") as f:
        #f.write(html_content)
        
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.body

    if body is None:
        return []

    chunks = []
    current_chunk = ""

    def add_chunk(chunk):
        clean_chunk = chunk.strip()
        if clean_chunk:
            chunks.append(clean_chunk)

    def is_semantic_split(tag: Tag):
        """Generic semantic split points without relying on class names."""
        return tag.name in ['section', 'article', 'main', 'header', 'div', 'footer', 'script', 'nav']

    for element in body.children:
        if isinstance(element, Tag):
            element_str = str(element)
            if is_semantic_split(element) or len(current_chunk) + len(element_str) > max_chunk_size:
                add_chunk(current_chunk)
                current_chunk = element_str
            else:
                current_chunk += element_str

    add_chunk(current_chunk)
    return chunks




def transform_new_website_chunk(body_doc, url, filename):
    final_chunks = []
    transformed_chunks = []

    final_chunks = extract_body_chunks(str(body_doc.page_content))
    folder_name = os.path.splitext(filename)[0]
    output_folder_path = create_output_folder_path(folder_name)
    chunk_indo_filename = f"chunks_info_{filename}.txt"
    save_chunks_to_file(output_folder_path, final_chunks, chunk_indo_filename)

    output_requirements = load_output_requirements("output_requirement")
    for index, chunk in enumerate(final_chunks):
        summary , components = summarize_html_chunk(chunk,ds_components)
        print("post summarize")
        design_standard_css = retrieve_design_standard(components, vectorstore_with_css)
        design_standard_yaml = retrieve_design_standard(components, vectorstore_with_yaml)

        formatted_prompt = create_transform_prompt(output_requirements, chunk, summary, design_standard_css, design_standard_yaml)
        response = llm.invoke(formatted_prompt).content.strip()

        match = re.search(r"```(?:html)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            response = match.group(1).strip()

        response = strip_outer_tags(response)
        

        # ------------------ Correction Loop Start ------------------
        correction_needed = 0
        correction_iteration = 0
        max_corrections = 2

        #Log Original Prompt
        chunk_filename = f"chunk_prompt_{index+1:02d}"
        chunk_filename_iter = f"{chunk_filename}_corrected_iter{correction_iteration}"
        save_prompt_response_to_file(output_folder_path, chunk_filename_iter, formatted_prompt, response, summary)

        while correction_needed and correction_iteration < max_corrections:
            correction_prompt = create_correction_prompt(formatted_prompt, response)
            correction_prompt_messages = [
                {"role": "system", "content": correction_prompt["system"]},
                {"role": "user", "content": correction_prompt["user"]}
            ]

         
            correction_response = llm.invoke(correction_prompt_messages).content.strip()
            corrected_output_requirements_match = re.search(r"## MANDATORY OUTPUT RULES:\s*(.*?)(?=\n##|\Z)", correction_response, re.DOTALL)

            corrected_output_requirements_str = ""
            if corrected_output_requirements_match:
                corrected_output_requirements_str = corrected_output_requirements_match.group(1).strip()
            else:
                print("Warning: corrected_output_requirements_match is None. Skipping correction update.")

            corrected_output_requirements_str = corrected_output_requirements_match.group(1).strip()
            
            correction_chunk_filename = f"correction_prompt_{index+1:02d}_iter{correction_iteration+1}"
            save_correction_prompt_response_to_file(output_folder_path, correction_chunk_filename, correction_prompt_messages, correction_response, corrected_output_requirements_str)

            if "# No Complaint Issue" in correction_response:
                correction_needed = False
                break

            
            if not corrected_output_requirements_match:
                break
            
            formatted_prompt = create_transform_prompt(corrected_output_requirements_str, chunk, summary, design_standard_css, design_standard_yaml)
            response = llm.invoke(formatted_prompt).content.strip()

            match = re.search(r"```(?:html)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
            if match:
                response = match.group(1).strip()

            response = strip_outer_tags(response)
        
            chunk_filename_iter = f"{chunk_filename}_corrected_iter{correction_iteration+1}"
            save_prompt_response_to_file(output_folder_path, chunk_filename_iter, formatted_prompt, response, summary)

            correction_iteration += 1
        # ------------------ Correction Loop End ------------------

        transformed_chunks.append(response)

    # Join final body content
    join_without_llm = 1
    if join_without_llm:
        final_body = "\n".join(f'<section class="py-1"><div class="container">\n{chunk}\n</div></section>' for chunk in transformed_chunks)
    else:
        join_prompt = create_join_prompt(transformed_chunks, filename)
        final_body = llm.invoke(join_prompt).content.strip()
        match = re.search(r"```(?:html)?\n(.*?)```", final_body, re.DOTALL | re.IGNORECASE)
        if match:
            final_body = match.group(1).strip()
    
    #original_html_body = extract_html_body(str(body_doc.page_content))
    #allignment_prompt = create_alignment_issue_prompt(
    #original_body=original_html_body,
    #transformed_body=final_body,
    #filename=chunk_filename,
    #output_dir= output_folder_path
#)
    #alignment_issues_response = llm.invoke(allignment_prompt).content.strip()
    #alignment_issues_filepath = os.path.join(output_folder_path, filename + "_alignment_issues.txt")
    #with open(alignment_issues_filepath, "w", encoding="utf-8") as f:
        #f.write(alignment_issues_response)
        
    custom_design_spec_css = load_custom_design_spec_css()
    custom_design_spec_js = load_custom_design_spec_js()

    final_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="poc-ds.css">
        <title>Transformed Website</title>
    </head>
    <body>
        {final_body}
    </body>
    </html>
    """

    
    body_output_file_path = os.path.join(output_folder_path, f"body_transformed_{filename}")
    with open(body_output_file_path, "w", encoding="utf-8") as f:
        f.write(final_body)

    css_file_path = os.path.join(OUTPUT_DIR, "poc-ds.css")
    final_html = inline_css_from_file(final_html, css_file_path)

    output_file_path = os.path.join(OUTPUT_DIR, f"transformed_{filename}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    st.subheader("Transformed Output")
    st.subheader("Generated custom_design_spec Output")
    st.session_state.custom_design_spec_code = final_html
    st.rerun()
    return final_html


# Streamlit UI Begin*************************************************************************************************************************
#st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top:2rem;
            padding-left:10px !important;
            padding-right:20px !important;
        }
        .st-emotion-cache-16tyu1 h1{
        padding-top:0px}
    </style>
""", unsafe_allow_html=True)

# Initializations
if "custom_design_spec_code" not in st.session_state:
    st.session_state.custom_design_spec_code = ""
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "isChunk" not in st.session_state:
    st.session_state.isChunk = False

def reset_response():
    st.session_state.custom_design_spec_code = ""
#------------------------------------------------------------------------------------------------
st.title("Transform to custom_design_spec design")

with st.sidebar:
    option = st.radio("Select an option:", ["Generate Components", "Create Website"], key="nav_option", on_change=reset_response)

if option == "Generate Components":
    toggle_top = st.toggle("Show Top", value=True)

    if toggle_top:
        st.subheader("Generate UI Components")
        if "component_requests" not in st.session_state:
            st.session_state.component_requests = []

        if st.button("Add Component"):
            st.session_state.component_requests.append({"type": None})

        for i, component in enumerate(st.session_state.component_requests):
            with st.expander(f"Component {i+1}"):
                component_type = st.selectbox(
                    "Select Component Type",
                    ["Button", "Radio Button", "Dropdown"],
                    key=f"component_type_{i}"
                )
                st.session_state.component_requests[i]["type"] = component_type

                if component_type == "Button":
                    button_type = st.selectbox(
                        "Select Button Type",
                        ["Primary", "Secondary", "Success", "Danger",
                        "Warning", "Info", "Light", "Dark", "Link"],
                        key=f"button_type_{i}"
                    )
                    st.session_state.component_requests[i]["details"] = {"button_type": button_type}

                elif component_type == "Radio Button":
                    num_radio = st.number_input("Number of Radio Buttons", min_value=1, max_value=10, value=2, key=f"num_radio_{i}")
                    radio_labels = [st.text_input(f"Radio Button {j+1} Label", key=f"radio_label_{i}_{j}") for j in range(num_radio)]
                    st.session_state.component_requests[i]["details"] = {"num_radio": num_radio, "radio_labels": radio_labels}

                elif component_type == "Dropdown":
                    num_dropdown = st.number_input("Number of Dropdown Options", min_value=1, max_value=10, value=2, key=f"num_dropdown_{i}")
                    dropdown_labels = [st.text_input(f"Dropdown Option {j+1} Label", key=f"dropdown_label_{i}_{j}") for j in range(num_dropdown)]
                    st.session_state.component_requests[i]["details"] = {"num_dropdown": num_dropdown, "dropdown_labels": dropdown_labels}

        if st.button("Generate Components"):
            prompts = []
            for component in st.session_state.component_requests:
                if component["type"] == "Button":
                    prompts.append(f"Generate a {component['details']['button_type'].lower()} button using custom_design_spec 5.")
                elif component["type"] == "Radio Button":
                    labels = ", ".join(component["details"]["radio_labels"])
                    prompts.append(f"Generate a radio button group with {component['details']['num_radio']} options: {labels} using custom_design_spec 5.")
                elif component["type"] == "Dropdown":
                    labels = ", ".join(component["details"]["dropdown_labels"])
                    prompts.append(f"Generate a dropdown with {component['details']['num_dropdown']} options: {labels} using custom_design_spec 5.")
            if prompts:
                final_prompt = "\n".join(prompts)
                generate_components(final_prompt)

if option == "Create Website":
    with st.expander("Create Website", expanded=False if st.session_state.custom_design_spec_code else True):
        st.subheader("Choose how to transform a web page")

        browse_option = st.selectbox("Browse Mode", ["Browse by URL", "Browse by File"])

        if browse_option == "Browse by URL":
            url_input = st.text_input("Enter a full webpage URL (e.g., https://example.com)")
            if url_input:
                if st.button("Transform from URL"):
                    with st.spinner("Fetching and transforming the webpage..."):
                        try:
                            doc = fetch_html_as_document(url_input)
                            body_doc = extract_body_only(doc)
                            if body_doc:
                                transform_new_website_chunk(body_doc,url_input ,filename=url_input.split("/")[-1] or "url_page.html")
                        except Exception as e:
                            st.error(f"Error processing URL: {e}")

        elif browse_option == "Browse by File":
            uploaded_files = st.file_uploader("Upload HTML, CSS, or JS files", type=["html", "css", "js"], accept_multiple_files=True)
            if uploaded_files:
                docs = []
                for file in uploaded_files:
                    try:
                        content = file.read().decode("utf-8")
                        doc = Document(page_content=content, metadata={"filename": file.name})
                        body_doc = extract_body_only(doc)
                        if body_doc:
                            docs.append(body_doc)
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")
                if docs:
                    st.success(f"Successfully extracted body content from {len(docs)} file(s)!")
                    if st.button("Transform from File(s)"):
                        with st.spinner("Processing..."):
                            for doc in docs:
                                try:
                                    transform_new_website_chunk(doc, filename=doc.metadata["filename"])
                                except Exception as e:
                                    st.error(f"Error processing {doc.metadata['filename']}: {e}")

if st.session_state.custom_design_spec_code:
    st.markdown("---------------")
    toggle = st.toggle("Show code", value=False)
    
    if not toggle:
        st.subheader("Live Preview")
        st.components.v1.html(st.session_state.custom_design_spec_code, height=500, scrolling=True)
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Live Preview")
            st.components.v1.html(st.session_state.custom_design_spec_code, height=500, scrolling=True)

        with col2:
            st.subheader("Code")
            if not st.session_state.edit_mode:
                if st.button("Edit", use_container_width=True):
                    st.session_state.edit_mode = True
                    st.rerun()
            else:
                edited = st.text_area("Edit code", value=st.session_state.custom_design_spec_code, height=500)
                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("save", use_container_width=True):
                        st.session_state.custom_design_spec_code = edited
                        st.session_state.edit_mode = False
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.edit_mode = False
                        st.rerun()

            if not st.session_state.edit_mode:
                st.code(st.session_state.custom_design_spec_code, language="html")


#************************************************************************************************************************************
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------
# 1. Transform from URL
# ----------------------
class URLInput(BaseModel):
    url: str

@app.post("/transform/url")
async def transform_from_url(data: URLInput):
    try:
        print("Request transform_from_url begin")
        doc = fetch_html_as_document(data.url)
        body_doc = extract_body_only(doc)
        if body_doc:
            transformed_html = transform_new_website_chunk(
                body_doc,
                data.url,
                filename=data.url.split("/")[-1] or "url_page.html"
            )
            print("Request transform_from_url end")
        else:
            transformed_html = "No body content found."
        
    except Exception as e:
        # Return the exception message as transformed_content
        return {"transformed_content": f"Error: {str(e)}"}

    print("Request transform_from_url end")
    return {"transformed_content": transformed_html}



# -----------------------
# 2. Transform from File
# -----------------------
@app.post("/transform/file")
async def transform_from_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    class Doc:
        def __init__(self, content, filename):
            self.content = content
            self.metadata = {"filename": filename}

    with open(temp_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = Doc(content, file.filename)
    result = transform_new_website_chunk(doc, filename=doc.metadata["filename"])

    os.remove(temp_path)
    return {"transformed_content": result}

# -------------------------------
# 3. Generate Components from Prompt
# -------------------------------


class PromptInput(BaseModel):
    prompt: List[str]

@app.post("/generate/components")
async def generate_from_prompt(data: PromptInput):
    result = generate_components(data.prompt)
    return {"generated_components": result}

