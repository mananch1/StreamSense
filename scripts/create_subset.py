"""
Memory-efficient two-pass parser for SNAP Amazon Movie Reviews (9.3GB).
Extracts the most-reviewed movie's reviews into CSV files for rapid streaming & ML experiments.
"""

import os
import sys
import csv
import json
import time
from collections import Counter
from datetime import datetime

# Potential dataset locations
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "data", "movies.txt"),
    os.path.join(os.path.dirname(__file__), "..", "movies.txt"),
]

def find_dataset_file():
    for p in DATA_PATHS:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p) and os.path.getsize(abs_p) > 1000:
            return abs_p
    raise FileNotFoundError("Could not find movies.txt in project root or data/ directory.")

def pass_1_find_top_movies(filepath, max_lines=None):
    """
    Pass 1: Stream line-by-line to count reviews per productId.
    Uses virtually zero memory (~Counter of strings).
    """
    print(f"[*] Pass 1: Scanning {filepath} to find most-reviewed movies...")
    start_time = time.time()
    counts = Counter()
    total_reviews = 0
    line_count = 0
    
    with open(filepath, "r", encoding="latin-1", errors="replace") as f:
        for line in f:
            line_count += 1
            if line.startswith("product/productId:"):
                asin = line[len("product/productId:"):].strip()
                if asin:
                    counts[asin] += 1
                    total_reviews += 1
            
            if total_reviews % 500000 == 0 and total_reviews > 0:
                elapsed = time.time() - start_time
                print(f"    Scanned {total_reviews:,} reviews ({line_count:,} lines) in {elapsed:.1f}s...")
            
            if max_lines and line_count >= max_lines:
                break
                
    elapsed = time.time() - start_time
    print(f"[+] Pass 1 complete! Total reviews: {total_reviews:,}, Unique movies: {len(counts):,} ({elapsed:.1f}s)")
    top_10 = counts.most_common(10)
    print("\nTop 10 Most-Reviewed Products/Movies:")
    for rank, (asin, cnt) in enumerate(top_10, 1):
        print(f"  {rank}. ASIN: {asin} -> {cnt:,} reviews")
        
    return top_10

def parse_record_block(block_lines):
    """
    Parse a block of lines representing one review.
    Handles multi-line fields cleanly.
    """
    record = {
        "productId": "",
        "userId": "",
        "profileName": "",
        "helpfulness": "",
        "score": "",
        "time": "",
        "summary": "",
        "text": ""
    }
    
    current_field = None
    text_lines = []
    
    for line in block_lines:
        line_clean = line.rstrip("\r\n")
        if line_clean.startswith("product/productId:"):
            current_field = "productId"
            record["productId"] = line_clean[18:].strip()
        elif line_clean.startswith("review/userId:"):
            current_field = "userId"
            record["userId"] = line_clean[14:].strip()
        elif line_clean.startswith("review/profileName:"):
            current_field = "profileName"
            record["profileName"] = line_clean[19:].strip()
        elif line_clean.startswith("review/helpfulness:"):
            current_field = "helpfulness"
            record["helpfulness"] = line_clean[19:].strip()
        elif line_clean.startswith("review/score:"):
            current_field = "score"
            record["score"] = line_clean[13:].strip()
        elif line_clean.startswith("review/time:"):
            current_field = "time"
            record["time"] = line_clean[12:].strip()
        elif line_clean.startswith("review/summary:"):
            current_field = "summary"
            record["summary"] = line_clean[15:].strip()
        elif line_clean.startswith("review/text:"):
            current_field = "text"
            text_lines = [line_clean[12:].strip()]
        else:
            if current_field == "text":
                text_lines.append(line_clean)
            elif current_field == "summary":
                record["summary"] += " " + line_clean
                
    record["text"] = " ".join(text_lines).strip()
    return record

def pass_2_extract_reviews(filepath, target_asin, out_csv_path, sample_csv_path, sample_size=500):
    """
    Pass 2: Extract all reviews for target_asin and write to CSV.
    """
    print(f"\n[*] Pass 2: Extracting reviews for target ASIN '{target_asin}' into {out_csv_path}...")
    start_time = time.time()
    
    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)
    
    extracted_records = []
    current_block = []
    total_lines = 0
    
    fieldnames = ["productId", "userId", "profileName", "helpfulness", "score", "time", "date", "summary", "text"]
    
    with open(filepath, "r", encoding="latin-1", errors="replace") as f_in:
        for line in f_in:
            total_lines += 1
            if line.startswith("product/productId:"):
                if current_block:
                    rec = parse_record_block(current_block)
                    if rec["productId"] == target_asin:
                        # Format timestamp to ISO date string
                        try:
                            ts = int(rec["time"])
                            rec["date"] = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            rec["date"] = ""
                        extracted_records.append(rec)
                    current_block = []
            current_block.append(line)
            
    # Process last block
    if current_block:
        rec = parse_record_block(current_block)
        if rec["productId"] == target_asin:
            try:
                ts = int(rec["time"])
                rec["date"] = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                rec["date"] = ""
            extracted_records.append(rec)
            
    # Sort chronologically by time
    def safe_time_sort(r):
        try:
            return int(r["time"])
        except Exception:
            return 0
            
    extracted_records.sort(key=safe_time_sort)
    
    # Write full movie reviews CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in extracted_records:
            writer.writerow(r)
            
    # Write sample CSV (500 reviews)
    sample_records = extracted_records[:sample_size]
    with open(sample_csv_path, "w", newline="", encoding="utf-8") as f_sample:
        writer = csv.DictWriter(f_sample, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample_records:
            writer.writerow(r)
            
    elapsed = time.time() - start_time
    print(f"[+] Extracted {len(extracted_records):,} reviews for {target_asin} in {elapsed:.1f}s!")
    print(f"[+] Saved full CSV to: {out_csv_path}")
    print(f"[+] Saved sample CSV ({len(sample_records)} reviews) to: {sample_csv_path}")
    
    # Calculate score distribution
    score_dist = Counter([r["score"] for r in extracted_records])
    print("\nScore Distribution:")
    for score in sorted(score_dist.keys()):
        print(f"  ★ {score}: {score_dist[score]} reviews ({score_dist[score]/len(extracted_records)*100:.1f}%)")
        
    meta = {
        "target_asin": target_asin,
        "total_reviews": len(extracted_records),
        "sample_size": len(sample_records),
        "score_distribution": dict(score_dist),
        "start_date": extracted_records[0]["date"] if extracted_records else "",
        "end_date": extracted_records[-1]["date"] if extracted_records else "",
        "generated_at": datetime.now().isoformat()
    }
    
    meta_path = os.path.join(os.path.dirname(out_csv_path), "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2)
    print(f"[+] Metadata written to: {meta_path}")

def main():
    try:
        data_file = find_dataset_file()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    top_10 = pass_1_find_top_movies(data_file)
    top_asin = top_10[0][0]
    
    out_csv = os.path.join(os.path.dirname(data_file), "top_movie_reviews.csv")
    sample_csv = os.path.join(os.path.dirname(data_file), "sample_reviews.csv")
    
    pass_2_extract_reviews(data_file, top_asin, out_csv, sample_csv, sample_size=500)
    print("\n[✓] Dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
