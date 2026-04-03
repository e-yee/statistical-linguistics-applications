import torch
import polars as pl

from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[3]
NOT_TRANSLATED_FILE = BASE_DIR / "" # Data path

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
        pl.Series(name="Translation", values=result)
    ).write_csv("translated.csv")

def translate_data(
    sentences: list[str], 
    tokenizer, 
    model
)-> list[str]:
    """Translate data.

    Parameters
    ----------
    sentences : list[str]
        A list of sentences to be translated.
    tokenizer : Any
        The translating model's tokenizer.
    model : Any
        The translating model.

    Returns
    -------
    list[str]
        A list of translated sentences. 
    """
    src_lang = "eng_Latn"
    tgt_lang = "vie_Latn"
    bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    batch_size = 64
    result = []
    
    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        batch = sentences[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            src_lang=src_lang
        ).to(model.device)

        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                forced_bos_token_id=bos_token_id,
                max_new_tokens=128
            )  
            
        outputs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        result.extend(outputs)
    
    return result

def main():
    # Load dataset
    df = load_data(NOT_TRANSLATED_FILE)
    sentences = df["Sentence"].to_list()
    print(df)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    torch.cuda.empty_cache()
    
    # Translate data
    result = translate_data(sentences, tokenizer, model)
    
    # Write data
    write_data(df, result)

if __name__ == "__main__":
    main()