import torch
import polars as pl

from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parents[3]
ALIGNED_FILE = BASE_DIR / "" # Data path

CACHE = {}
MODEL = SentenceTransformer('sentence-transformers/LaBSE')

def get_embeddings(words: list[str]) -> torch.Tensor:
    """Get embeddings of words.

    Parameters
    ----------
    words : list[str]
        A list of word to be embedded.

    Returns
    -------
    torch.Tensor
        A tensor containing the embeddings of the input words.
    """
    new_words = [w for w in words if w not in CACHE]
    
    if new_words:
        embs = MODEL.encode(new_words, convert_to_tensor=True)
        CACHE.update(dict(zip(new_words, embs)))
    
    return torch.vstack([CACHE[w] for w in words])

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

def write_data(
    df: pl.DataFrame, 
    result: list[str]
) -> None:
    """Write a DataFrame to a file.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to be written.
    result : list[str]
        A list of values used to create or replace a column in the DataFrame.
    """
    df.with_columns(
        pl.Series(name="ViEnAlignments", values=result)
    ).write_csv("not_sensed.csv")
    
def refined_align(df: pl.DataFrame) -> list[str]:
    """Refine the data alignments.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame contained alignments.

    Returns
    -------
    list[str]
        A list of refined alignments.
    """
    lines = []

    for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Refining..."):
        en_words = row["Sentence"].split()
        vi_words = row["Text"].split()
        pairs = row["ViEnAlignments"].split(";")

        en_list, vi_list, idx_map = [], [], []

        for j, pair in enumerate(pairs):
            vi, en = map(int, pair.split("-"))

            en_word = en_words[en].lower().replace("<", "").replace(">", "").replace("_", " ")
            vi_word = vi_words[vi].lower().replace("<", "").replace(">", "").replace("_", " ")

            en_list.append(en_word)
            vi_list.append(vi_word)
            idx_map.append(j)

        if en_list:
            emb_en = get_embeddings(en_list)
            emb_vi = get_embeddings(vi_list)

            sims = util.cos_sim(emb_en, emb_vi).diagonal()

            for k, sim in enumerate(sims):
                if sim.item() < 0.67:
                    pairs[idx_map[k]] = f"{pairs[idx_map[k]].split('-')[0]}-null"

        pairs = sorted(pairs, key=lambda x: int(x.split("-")[0]))
        indices = torch.zeros(len(vi_words), dtype=int)
        
        for pair in pairs:
            vi, en = pair.split("-")
            indices[int(vi)] = int(en) + 1 if en is not "null" else 0
            
        lines.append(str(indices.tolist()))
    
    return lines

def main():
    # Load dataset
    df = load_data(ALIGNED_FILE)
    print(df)
    
    # Translate data
    lines = refined_align(df)
    
    # Write data
    write_data(df, lines)
    
if __name__ == "__main__":
    main()