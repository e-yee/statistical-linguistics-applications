import ast
import torch
import polars as pl

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[3]
NOT_SENSE_TAGGED_FILE = BASE_DIR / "" # Data path
INSTANCES_FILE = BASE_DIR / "" # Data path
HOANGPHE_D_FILE = BASE_DIR / "" # Data path

CACHE = {}
BATCH_CACHE = {}
MODEL = SentenceTransformer('sentence-transformers/LaBSE')

def get_embeddings(definitions: list[str]) -> torch.Tensor:
    """Get embedding matrices of definitions.

    Parameters
    ----------
    definitions : list[str]
        A list of definitions.

    Returns
    -------
    torch.Tensor
        A tensor of result embeddings.
    """
    key = tuple(definitions)

    if key in BATCH_CACHE:
        return BATCH_CACHE[key]
        
    missing = [d for d in definitions if d not in CACHE]
    
    if missing:
        embs = MODEL.encode(missing, convert_to_tensor=True, normalize_embeddings=True)
        CACHE.update(dict(zip(missing, embs)))

    result = torch.vstack([CACHE[d] for d in definitions])
    
    BATCH_CACHE[key] = result
    return result


def get_instance_indices(words: list[str]) -> list[int]:
    """Get instance indices from english words.

    Parameters
    ----------
    words : list[str]
        English words.

    Returns
    -------
    list[int]
        A list of instance indices.
    """    
    return [i for i, w in enumerate(words) if "<" in w]

def load_data(path: Path) -> pl.DataFrame:
    """Load DataFrame.

    Parameters
    ----------
    path : Path
        The path to data.

    Returns
    -------
    pl.DataFrame
        The loaded DataFrame.
    """
    return pl.read_csv(path)

def write_data(result: list[str]) -> None:
    """Write a DataFrame to a file.

    Parameters
    ----------
    result : list[str]
        A list of values.
    """
    pl.from_dicts(result).write_csv("sensed.csv")

def extract_aligned_instances(
    nsn_df: pl.DataFrame, 
    hpd_lookup: defaultdict[list], 
    isc_lookup: defaultdict[list]
) -> list[dict]:
    """Extract aligned instances for not sense tagged dataframe.

    Parameters
    ----------
    nsn_df : pl.DataFrame
        Not sense tagged DataFrame.
    hpd_lookup : defaultdict[list]
        Hoang Phe nouns lookup table.
    isc_lookup : defaultdict[list]
        SemCor instances lookup table

    Returns
    -------
    list[dict]
        A list of sense tagged instances.
    """
    lines = []
    
    for row in tqdm(nsn_df.iter_rows(named=True), total=nsn_df.height, desc="Sense tagging..."):
        sentence_id = row["SentenceId"]
        
        en_sentence = row["Sentence"].split()
        vi_sentence = row["Text"].split()
        alignments = ast.literal_eval(row["ViEnAlignments"])

        instance_ids = get_instance_indices(en_sentence)
        instance_id_map = {idx: i for i, idx in enumerate(instance_ids)}

        isc_count = 0
        
        for i, a in enumerate(alignments):
            if a == 0:
                continue
                
            en_idx = a - 1
            
            if (
                en_idx not in instance_ids 
                or "<" not in en_sentence[en_idx] 
                or "<" not in vi_sentence[i]
            ):
                continue

            vi_word = vi_sentence[i].lower().replace("_", " ").replace("<", "").replace(">", "")
            
            instance_id = f"{sentence_id}.t{instance_id_map[en_idx]:03d}"
            
            vi_hpd = hpd_lookup[vi_word]
            en_isc = isc_lookup[instance_id]

            vi_embs = get_embeddings(vi_hpd)
            en_embs = get_embeddings(en_isc)
            similarities = vi_embs @ en_embs.T
            
            flat_idx = similarities.argmax()
            r = flat_idx // similarities.shape[1]
            c = flat_idx % similarities.shape[1]
            
            lines.append({
                "InstanceId": f"{sentence_id}.t{isc_count:03d}",
                "Word": vi_word,
                "SenseId": f"{vi_word.replace(" ", "_")}.d.{r + 1:02d}",
                "Definition": vi_hpd[r]
            })

            isc_count += 1

def main():
    # Load data
    hpd_df = load_data(HOANGPHE_D_FILE)
    print(hpd_df)
    
    isc_df = load_data(INSTANCES_FILE)
    print(isc_df)
    
    nsn_df = load_data(NOT_SENSE_TAGGED_FILE)
    print(nsn_df)

    # Init lookup table
    hpd_lookup = defaultdict(list)
    for row in hpd_df.iter_rows(named=True):
        hpd_lookup[row["Word"]].append(row["Meaning"])

    isc_lookup = defaultdict(list)
    for row in isc_df.to_dicts():
        isc_lookup[row["InstanceId"]].append(row["Definition"])
    
    # Extract instances
    result = extract_aligned_instances(nsn_df, hpd_lookup, isc_lookup)
     
    # Write data   
    write_data(result)

if __name__ == "__main__":
    main()