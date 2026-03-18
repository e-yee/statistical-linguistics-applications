import torch
import polars as pl
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[3]
NOT_TRANSLATED_FILE = BASE_DIR / "" # Data path

def translate_data():
    # Load dataset
    df =  pl.read_csv(NOT_TRANSLATED_FILE)
    print(df)
    
    sentences = df["Sentence"].to_list()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Model's settings
    src_lang = "eng_Latn"
    tgt_lang = "vie_Latn"
    bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    batch_size = 64
    result = []
    
    torch.cuda.empty_cache()
    
    # Translate data
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
    
    # Write data to file
    df = df.with_columns(
        pl.Series(name="Translation", values=result)
    ).write_csv("translated.csv")
       
def main():
    translate_data()

if __name__ == "__main__":
    main()