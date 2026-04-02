import torch
import polars as pl

from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parents[3]
ALIGNED_FILE = BASE_DIR / "" # Data path

CACHE = {}

def get_embeddings(
    words: list[str], 
    model
) -> torch.Tensor:
    """Get embeddings of words.

    Parameters
    ----------
    words : list[str]
        A list of word to be embedded.
    model : Any
        The embedding model.

    Returns
    -------
    torch.Tensor
        A tensor containing the embeddings of the input words.
    """
    new_words = [w for w in words if w not in CACHE]
    
    if new_words:
        embs = model.encode(new_words, convert_to_tensor=True)
        for w, e in zip(new_words, embs):
            CACHE[w] = e
    
    return torch.stack([CACHE[w] for w in words])

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
    df = df.with_columns(
        pl.Series(name="ViEnAlignments", values=result)
    ).write_csv("not_sensed.csv")

    
def refined_align(
    df: pl.DataFrame, 
    model
) -> list[str]:
    """Refine the data alignments.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame contained alignments.
    model : Any
        The embedding model.

    Returns
    -------
    list[str]
        A list of refined alignments.
    """
    lines = []

    for i, row in tqdm(enumerate(df.iter_rows(named=True)), total=df.height):
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
            emb_en = get_embeddings(en_list, model)
            emb_vi = get_embeddings(vi_list, model)

            sims = util.cos_sim(emb_en, emb_vi).diagonal()

            for k, sim in enumerate(sims):
                if sim.item() < 0.67:
                    pairs[idx_map[k]] = f"{pairs[idx_map[k]].split('-')[0]}-null"

        lines.append(";".join(pairs))
    
    return lines

def main():
    # Load dataset
    df = load_data(ALIGNED_FILE)
    print(df)
    
    # Load model    
    model = SentenceTransformer('sentence-transformers/LaBSE')

    # Translate data
    lines = refined_align(df, model)
    
    # Write data
    write_data(df, lines)
    
if __name__ == "__main__":
    main()