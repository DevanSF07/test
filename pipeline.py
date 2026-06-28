#!/usr/bin/env python3
import os
import re
import sys
import json
import math
import argparse
import pypdf
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Import domain scoring
from uav_dictionary import calculate_domain_score

# Custom general/academic stopwords to filter out from candidate technical terms
GENERAL_STOPWORDS = {
    # NLTK English stopwords will be added dynamically, but here are extra general ones:
    "result", "results", "analysis", "analyses", "system", "systems", "process", "processes",
    "approach", "approaches", "chapter", "chapters", "section", "sections", "page", "pages",
    "table", "tables", "figure", "figures", "use", "uses", "value", "values", "data", "datum",
    "test", "tests", "time", "times", "user", "users", "number", "numbers", "level", "levels",
    "type", "types", "way", "ways", "case", "cases", "model", "models", "problem", "problems",
    "solution", "solutions", "etc", "eg", "ie", "et", "al", "example", "examples", "method",
    "methods", "performance", "performances", "comparison", "comparisons", "description",
    "descriptions", "concept", "concepts", "design", "designs", "development", "developments",
    "sizing", "criteria", "criterion", "factor", "factors", "parameter", "parameters",
    "requirement", "requirements", "study", "studies", "application", "applications",
    "author", "authors", "university", "department", "school", "press", "media", "science",
    "netherlands", "dordrecht", "springer", "editor", "editors", "isbn", "doi", "copyright"
}

def replace_ligatures(text):
    """
    Replaces common PDF extraction ligatures and curly quotes with standard ASCII characters.
    """
    ligatures = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "’": "'",
        "“": '"',
        "”": '"'
    }
    for lig, rep in ligatures.items():
        text = text.replace(lig, rep)
    return text

def clean_term(term_text):
    """
    Cleans candidate term text by removing leading/trailing punctuation,
    standardizing whitespace, and filtering out empty or invalid candidates.
    """
    # Replace newlines and multiple spaces
    term_text = re.sub(r'\s+', ' ', term_text)
    # Remove leading/trailing non-alphanumeric chars (keep hyphens and slashes inside words)
    term_text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', term_text)
    term_text = term_text.strip()
    
    # Filter out terms that are purely numeric, empty, or too short
    if not term_text or len(term_text) <= 2 or term_text.replace('.', '').replace('-', '').replace('/', '').isdigit():
        return ""
        
    return term_text

def extract_pdf_content(pdf_path, start_page_num, end_page_num, use_printed_pages=True):
    """
    Reads the PDF and extracts pages. If use_printed_pages is True, scans for printed
    page numbers start_page_num and end_page_num. Otherwise, uses physical page indices.
    """
    print(f"Opening PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total physical pages in PDF: {total_pages}")
    
    start_idx = None
    end_idx = None
    
    if use_printed_pages:
        print(f"Scanning for printed page numbers from {start_page_num} to {end_page_num}...")
        # Scan header and footer of pages to find the printed page numbers
        # We start scanning from page 150 to 300 to speed up search for printed pages around 160-169
        scan_range = range(max(0, start_page_num - 20), min(total_pages, end_page_num + 300))
        for idx in scan_range:
            text = reader.pages[idx].extract_text()
            if not text:
                continue
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if not lines:
                continue
                
            first_line = lines[0]
            last_line = lines[-1] if len(lines) > 1 else ""
            
            def check_match(line, target_num):
                if not line:
                    return False
                tokens = [t.strip("-,.()\"'") for t in line.split()]
                if not tokens:
                    return False
                # Page number is typically the first or last token in running headers or footers
                return tokens[0] == str(target_num) or tokens[-1] == str(target_num)
                
            if check_match(first_line, start_page_num) or check_match(last_line, start_page_num):
                if start_idx is None:
                    start_idx = idx
                    print(f"  Found printed page {start_page_num} at physical page {idx + 1}")
                    
            if check_match(first_line, end_page_num) or check_match(last_line, end_page_num):
                end_idx = idx
                
        if start_idx is not None and end_idx is not None:
            print(f"Mapped printed pages [{start_page_num}-{end_page_num}] to physical pages [{start_idx + 1}-{end_idx + 1}] (indices {start_idx}-{end_idx})")
        else:
            if start_idx is None:
                print(f"  Warning: Could not find printed page {start_page_num}")
            if end_idx is None:
                print(f"  Warning: Could not find printed page {end_page_num}")
            print("Falling back to physical page numbers.")
            
    # Fallback to physical pages (1-indexed input)
    if start_idx is None or end_idx is None:
        start_idx = start_page_num - 1
        end_idx = end_page_num - 1
        print(f"Using physical pages [{start_idx + 1}-{end_idx + 1}]")
        
    # Bounds check
    start_idx = max(0, min(start_idx, total_pages - 1))
    end_idx = max(0, min(end_idx, total_pages - 1))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
        
    extracted_content = []
    print(f"Extracting text from physical page {start_idx + 1} to {end_idx + 1}...")
    for idx in range(start_idx, end_idx + 1):
        page_text = reader.pages[idx].extract_text()
        if page_text:
            extracted_content.append(f"\n--- PHYSICAL PAGE {idx + 1} (PRINTED PAGE ESTIMATE) ---\n")
            extracted_content.append(page_text)
            
    return "".join(extracted_content)

def extract_technical_terms(text):
    """
    Parses text using NLP techniques to extract candidate technical terms (Noun Phrases)
    and filters/scores them using the UAV domain dictionary.
    """
    print("Initializing NLP resources...")
    # Get NLTK English stopwords
    stop_words = set(stopwords.words('english'))
    # Combine with our custom academic stopwords
    all_stopwords = stop_words.union(GENERAL_STOPWORDS)
    
    # Clean text: replace ligatures and remove hyphen line breaks
    cleaned_text = replace_ligatures(text)
    cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)
    print(f"Segmented text into {len(sentences)} sentences.")
    
    # Define a Noun Phrase chunker grammar:
    # A technical term is typically:
    # - Optional adjectives/nouns, followed by a sequence of one or more nouns.
    # - E.g. "small UAV design", "gimbal camera", "rechargeable battery"
    grammar = r"""
        NP: {<JJ|JJR|JJS|NN|NNS|NNP|NNPS>*<NN|NNS|NNP|NNPS>+}
    """
    chunk_parser = nltk.RegexpParser(grammar)
    
    # Map from lower_case_term -> {"count": count, "original_cases": {case: count}}
    term_aggregation = {}
    
    print("Parsing sentences for noun phrase candidate terms...")
    for sent in sentences:
        words = word_tokenize(sent)
        if not words:
            continue
        try:
            tagged = pos_tag(words)
        except Exception as e:
            # Fallback if POS tagging has issues
            continue
            
        tree = chunk_parser.parse(tagged)
        
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            # Reconstruct the NP string
            np_words = [word for word, tag in subtree.leaves()]
            
            # Skip single characters or single letters
            if len(np_words) == 1 and len(np_words[0]) <= 2:
                continue
                
            # Stopword filtering for the candidate term
            # If the term is a single word, it must not be a stopword
            if len(np_words) == 1:
                w_lower = np_words[0].lower()
                if w_lower in all_stopwords or w_lower.replace('-', '') in all_stopwords:
                    continue
            else:
                # For multi-word terms, if first or last word is a stopword/preposition, filter it
                if np_words[0].lower() in all_stopwords or np_words[-1].lower() in all_stopwords:
                    continue
                    
            candidate = clean_term(" ".join(np_words))
            if not candidate:
                continue
                
            # Filter if all words in candidate are generic stopwords
            cand_words = candidate.lower().split()
            if all(w in all_stopwords for w in cand_words):
                continue
                
            term_key = candidate.lower()
            if term_key not in term_aggregation:
                term_aggregation[term_key] = {"count": 0, "original_cases": {}}
            
            term_aggregation[term_key]["count"] += 1
            term_aggregation[term_key]["original_cases"][candidate] = term_aggregation[term_key]["original_cases"].get(candidate, 0) + 1
            
    print(f"Extracted {len(term_aggregation)} unique consolidated candidate terms.")
    
    # Score and filter candidate terms using UAV domain vocab
    processed_terms = []
    for term_lower, info in term_aggregation.items():
        count = info["count"]
        # Find the most frequent casing variant to use as the canonical form
        best_case = max(info["original_cases"], key=info["original_cases"].get)
        
        domain_score = calculate_domain_score(best_case)
        
        # Only retain terms that have non-zero relevance to the UAV domain
        if domain_score > 0:
            # Composite score combines domain relevance and frequency weight
            composite_score = round(domain_score * (1.0 + math.log1p(count)), 3)
            processed_terms.append({
                "term": best_case,
                "frequency": count,
                "domain_score": domain_score,
                "composite_score": composite_score
            })
            
    # Sort terms by composite score (highest first), then frequency, then alphabetically
    processed_terms.sort(key=lambda x: (x["composite_score"], x["frequency"], x["term"].lower()), reverse=True)
    
    return processed_terms

def main():
    parser = argparse.ArgumentParser(description="UAV PDF Page and Technical Term Extraction Pipeline")
    parser.add_argument("--pdf", type=str, default="/Users/devansinghfaujdar/Downloads/Handbook of Unmanned Aerial Vehicles-Springer Netherlands (2015).pdf",
                        help="Path to the PDF file")
    parser.add_argument("--start-page", type=int, default=160,
                        help="Start page number (defaults to printed page 160)")
    parser.add_argument("--end-page", type=int, default=169,
                        help="End page number (defaults to printed page 169)")
    parser.add_argument("--physical-pages", action="store_true",
                        help="Treat start-page and end-page as physical page numbers rather than printed page numbers")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save extracted outputs")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("UAV PIPELINE: EXTRACTING CONTENT AND TECHNICAL TERMS")
    print("="*60)
    
    try:
        # Step 1: Extract Text
        use_printed = not args.physical_pages
        extracted_text = extract_pdf_content(args.pdf, args.start_page, args.end_page, use_printed_pages=use_printed)
        
        text_out_path = os.path.join(args.output_dir, "extracted_text.txt")
        with open(text_out_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"\n[Success] Extracted raw text saved to: {text_out_path} ({len(extracted_text)} characters)")
        
        # Step 2: Extract Terms
        terms = extract_technical_terms(extracted_text)
        
        # Save JSON output
        json_out_path = os.path.join(args.output_dir, "extracted_terms.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(terms, f, indent=2)
        print(f"[Success] Extracted terms JSON report saved to: {json_out_path}")
        
        # Save Text output (Human Readable Report)
        txt_out_path = os.path.join(args.output_dir, "extracted_terms.txt")
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"UAV TECHNICAL TERMS EXTRACTED FROM PRINTED PAGES {args.start_page}-{args.end_page}\n")
            f.write(f"Total Unique Domain Terms Extracted: {len(terms)}\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'No.':<4} | {'Technical Term':<35} | {'Freq.':<5} | {'Domain Score':<12} | {'Composite Score':<15}\n")
            f.write("-" * 80 + "\n")
            for idx, item in enumerate(terms, 1):
                f.write(f"{idx:<4} | {item['term']:<35} | {item['frequency']:<5} | {item['domain_score']:<12.3f} | {item['composite_score']:<15.3f}\n")
                
        print(f"[Success] Extracted terms human-readable report saved to: {txt_out_path}")
        
        # Step 3: Extract Triplets
        print("\nStep 3: Extracting relationship triplets...")
        import subprocess
        triplets_script = os.path.join(os.path.dirname(__file__), "extract_triplets.py")
        subprocess.run([sys.executable, triplets_script], check=True)
        
        print(f"\nPipeline finished successfully. Extracted {len(terms)} technical terms and generated triplets.")
        print("="*60)
        
    except Exception as e:
        print(f"\nError executing pipeline: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
