#!/usr/bin/env python3
import os
import re
import csv
import sys

def clean_term(term):
    if not term:
        return ""
    # Remove leading numbers (like row numbers "139 mini" -> "mini")
    term = re.sub(r'^\d+\s*', '', term)
    # Remove leading/trailing non-alphanumeric chars
    term = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', term)
    term = term.strip().lower()
    # Normalize OCR errors
    replacements = {
        "classi": "class i",
        "rра": "rpa",
        "rрas": "rpas",
        "rрv": "rpv",
        "remntelvnilated": "remotely piloted",
        "ic-a": "is-a",
        "performs n": "performs",
        "unmanned aerial vehicle": "uav",
        "aerialvehicle": "aerial vehicle",
        "globalhawk": "global hawk",
        "pipeline inspectioh": "pipeline inspection",
        "suas": "small uas",
        "smalluavs": "small uavs"
    }
    for k, v in replacements.items():
        if term == k:
            term = v
    return term

def parse_ocr_ground_truth(ocr_file_path):
    ground_truth = set()
    if not os.path.exists(ocr_file_path):
        return ground_truth
        
    with open(ocr_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "ROW:" not in line:
                continue
            # Remove header prefix
            content = line.split("ROW:", 1)[1].strip()
            
            # Identify columns
            # The row typically uses '|' as column separator
            parts = [p.strip() for p in content.split("|")]
            
            # Filter out non-triplet lines like column headers or excel UI text
            if len(parts) < 3:
                continue
                
            subj = clean_term(parts[0])
            pred = clean_term(parts[1])
            obj = clean_term(parts[2])
            
            # Skip header row or UI rows
            if subj in ["subject", "alignment", "close"] or pred in ["predicate", "alignment", "number"]:
                continue
                
            # Basic validation
            if subj and pred and obj and len(subj) > 1 and len(obj) > 1:
                # Map various OCR variations of "is-a"
                if "is-a" in pred or "isa" in pred or "is a" in pred or "ic-a" in pred:
                    pred = "is-a"
                elif "performs" in pred:
                    pred = "performs"
                ground_truth.add((subj, pred, obj))
                
    return ground_truth

def read_extracted_triplets(csv_path):
    extracted = set()
    if not os.path.exists(csv_path):
        return extracted
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None) # skip header
        for row in reader:
            if len(row) >= 3:
                subj = clean_term(row[0])
                pred = clean_term(row[1])
                obj = clean_term(row[2])
                if subj and pred and obj:
                    extracted.add((subj, pred, obj))
    return extracted

def calculate_metrics(gt_set, ext_set):
    # Exact Match calculations
    tp_exact = gt_set.intersection(ext_set)
    fp_exact = ext_set - gt_set
    fn_exact = gt_set - ext_set
    
    precision_exact = len(tp_exact) / len(ext_set) if ext_set else 0
    recall_exact = len(tp_exact) / len(gt_set) if gt_set else 0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) else 0
    
    # Fuzzy Match calculations (to handle minor OCR typos or slight synonym differences)
    tp_fuzzy = []
    fp_fuzzy = []
    
    # Helper to check if a triplet matches fuzzily
    # Fuzzy matches if: subject and object match closely (e.g. subset or substring match) and predicate is same
    def is_fuzzy_match(t_ext, t_gt):
        s_ext, p_ext, o_ext = t_ext
        s_gt, p_gt, o_gt = t_gt
        
        if p_ext != p_gt:
            return False
            
        # Check subject similarity
        s_match = (s_ext in s_gt) or (s_gt in s_ext) or (s_ext.replace(" ", "") == s_gt.replace(" ", ""))
        # Check object similarity
        o_match = (o_ext in o_gt) or (o_gt in o_ext) or (o_ext.replace(" ", "") == o_gt.replace(" ", ""))
        
        return s_match and o_match

    matched_gt = set()
    for t_ext in ext_set:
        found = False
        for t_gt in gt_set:
            if is_fuzzy_match(t_ext, t_gt):
                tp_fuzzy.append((t_ext, t_gt))
                matched_gt.add(t_gt)
                found = True
                break
        if not found:
            fp_fuzzy.append(t_ext)
            
    fn_fuzzy = list(gt_set - matched_gt)
    
    precision_fuzzy = len(tp_fuzzy) / len(ext_set) if ext_set else 0
    recall_fuzzy = len(tp_fuzzy) / len(gt_set) if gt_set else 0
    f1_fuzzy = 2 * precision_fuzzy * recall_fuzzy / (precision_fuzzy + recall_fuzzy) if (precision_fuzzy + recall_fuzzy) else 0

    return {
        "gt_count": len(gt_set),
        "ext_count": len(ext_set),
        
        "tp_exact": list(tp_exact),
        "fp_exact": list(fp_exact),
        "fn_exact": list(fn_exact),
        "precision_exact": precision_exact,
        "recall_exact": recall_exact,
        "f1_exact": f1_exact,
        
        "tp_fuzzy": tp_fuzzy,
        "fp_fuzzy": fp_fuzzy,
        "fn_fuzzy": fn_fuzzy,
        "precision_fuzzy": precision_fuzzy,
        "recall_fuzzy": recall_fuzzy,
        "f1_fuzzy": f1_fuzzy
    }

def main():
    ocr_file = "output/ocr_raw.txt"
    csv_file = "output/extracted_triplets.csv"
    
    gt_set = parse_ocr_ground_truth(ocr_file)
    ext_set = read_extracted_triplets(csv_file)
    
    if not gt_set:
        print("Error: Ground truth set is empty. Check output/ocr_raw.txt")
        sys.exit(1)
        
    metrics = calculate_metrics(gt_set, ext_set)
    
    # Save Report to file
    report_path = "/Users/devansinghfaujdar/.gemini/antigravity/brain/09ec76e3-f864-45fe-ba9f-552ae57bff30/accuracy_metrics.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# UAV Triplet Extraction Accuracy Metrics Report\n\n")
        
        f.write("This report evaluates the accuracy of the relationship triplet extraction pipeline against the ground truth taxonomy extracted from the screenshots in the `Documents/groundtruth` directory.\n\n")
        
        f.write("## Overview Metrics\n\n")
        f.write(f"- **Total Ground Truth Triplets (from OCR)**: {metrics['gt_count']}\n")
        f.write(f"- **Total Extracted Triplets by Pipeline**: {metrics['ext_count']}\n\n")
        
        f.write("### 1. Exact Match Evaluation\n")
        f.write("Requires exact string character matching after basic cleaning:\n\n")
        f.write(f"- **True Positives (TP)**: {len(metrics['tp_exact'])}\n")
        f.write(f"- **False Positives (FP)**: {len(metrics['fp_exact'])}\n")
        f.write(f"- **False Negatives (FN)**: {len(metrics['fn_exact'])}\n")
        f.write(f"- **Precision**: {metrics['precision_exact']:.3f}\n")
        f.write(f"- **Recall**: {metrics['recall_exact']:.3f}\n")
        f.write(f"- **F1-Score**: {metrics['f1_exact']:.3f}\n\n")
        
        f.write("### 2. Fuzzy Match Evaluation (Recommended)\n")
        f.write("Accounts for minor OCR character typos (e.g. `remntelvnilated` for `remotely piloted`) and substring matches:\n\n")
        f.write(f"- **True Positives (TP)**: {len(metrics['tp_fuzzy'])}\n")
        f.write(f"- **False Positives (FP)**: {len(metrics['fp_fuzzy'])}\n")
        f.write(f"- **False Negatives (FN)**: {len(metrics['fn_fuzzy'])}\n")
        f.write(f"- **Precision**: {metrics['precision_fuzzy']:.3f}\n")
        f.write(f"- **Recall**: {metrics['recall_fuzzy']:.3f}\n")
        f.write(f"- **F1-Score**: {metrics['f1_fuzzy']:.3f}\n\n")
        
        f.write("## Sample True Positives (Matches)\n\n")
        f.write("| No. | Pipeline Triplet | Ground Truth Triplet (OCR) |\n")
        f.write("| :--- | :--- | :--- |\n")
        for idx, (ext, gt) in enumerate(metrics["tp_fuzzy"][:25], 1):
            f.write(f"| {idx} | `({ext[0]}, {ext[1]}, {ext[2]})` | `({gt[0]}, {gt[1]}, {gt[2]})` |\n")
            
        f.write("\n## Sample False Positives (Extra Extracted Triplets)\n\n")
        f.write("Triplets extracted by the pipeline but not listed in the ground truth sheet:\n\n")
        for idx, ext in enumerate(metrics["fp_fuzzy"][:15], 1):
            f.write(f"{idx}. `({ext[0]}, {ext[1]}, {ext[2]})`\n")
            
        f.write("\n## Sample False Negatives (Missing Triplets)\n\n")
        f.write("Triplets in the ground truth sheet that the pipeline did not extract (often due to being in other pages of the book, or absent from the physical pages 160-169 text):\n\n")
        for idx, gt in enumerate(metrics["fn_fuzzy"][:15], 1):
            f.write(f"{idx}. `({gt[0]}, {gt[1]}, {gt[2]})`\n")
            
    print(f"Metrics calculation complete. Report saved to: {report_path}")

if __name__ == "__main__":
    main()
