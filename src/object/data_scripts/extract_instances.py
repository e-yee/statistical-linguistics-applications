import polars as pl

from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[3]
NOT_SENSE_TAGGED_FILE = BASE_DIR / "" # Data path
INSTANCES_FILE = BASE_DIR / "" # Data path

def get_instance_indices(text: str) -> list[int]:
    words = text.split()
    indices = []
    
    for i, word in enumerate(words):
        if "|" in word:
            indices.append(i)
            
    return indices

def get_aligned_indices(alignments: str) -> list[tuple[int, int]]:
    pairs = alignments.split(";")
    indices = []
    
    for pair in pairs:
        vi, en = map(int, pair.split("-"))
        indices.append((vi, en))
        
    return indices

def extract_aligned_instances():
    # Load dataset
    df = pl.read_csv(NOT_SENSE_TAGGED_FILE)
    print(df)
    
    # Extract instances
    lines = []

    for row in df.iter_rows(named=True):
        en_instances_indices = get_instance_indices(row["Sentence"])
        lemma_instances_indices = get_instance_indices(row["Lemma"])    
        aligned_indices = get_aligned_indices(row["ViEnAlignments"])
        
        en_words = row["Sentence"].split()
        vi_words = row["Translation"].split()
        lemma_words = row["Lemma"].split()
        words_pos = row["POS"].split()
        
        for vi_index, en_index in aligned_indices:
            if en_index not in en_instances_indices:
                continue
            
            instance_index = en_instances_indices.index(en_index)
            instance_id = f"{row["SentenceId"]}.t{instance_index:03d}"
            
            if len(row["SentenceId"]) != 9:
                instance_id = f"{row["SentenceId"]}.t{instance_index:06d}"
            
            en_word = f"{en_words[en_index].replace("|", "")}"
            lemma_index = lemma_instances_indices[instance_index]
            lemma = f"{lemma_words[lemma_index].replace("|", "")}"
            pos = words_pos[lemma_index]
            vi_word = vi_words[vi_index]
            
            lines.append({
                "InstanceId": instance_id,
                "EnglishWord": en_word,
                "Lemma": lemma,
                "POS": pos,
                "VietnameseWord": vi_word
            })        

    lines.sort(key=lambda x: x["InstanceId"])
    
    # Extract aligned instances
    aligned_instances_lf = pl.from_dicts(data=lines).lazy()
    instances_lf = (
        pl.scan_csv(INSTANCES_FILE)
        .join(
            aligned_instances_lf
            .select("InstanceId")
            .unique(), on="InstanceId", how="inner"
        )
    )

    (pl.concat([aligned_instances_lf, instances_lf], how="align_left")
        .select([
            "InstanceId",
            "EnglishWord",
            "Lemma",
            "SenseId",
            "SynsetId",
            "VietnameseWord"
        ])
        .collect()
    ).write_csv("aligned_instances.csv")

def main():
    extract_aligned_instances()

if __name__ == "__main__":
    main()