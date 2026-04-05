import os
import re
import csv
import xml.etree.ElementTree as ET

from pathlib import Path
from nltk.corpus import wordnet as wn

BASE_DIR = Path(__file__).resolve().parents[3]
DICTIONARIES_DIR = BASE_DIR / "data" / "dictionaries"

VIETDICT_DIR = DICTIONARIES_DIR / "hoang_phe"

WSD_DIR = BASE_DIR / "data" / "wsd"
EVAL_DIR = WSD_DIR / "WSD_Evaluation_Framework"

class XMLToCSVConverter:
    @staticmethod
    def __convert_vietdict():
        """
        Convert Hoang Phe XML to CSV.

        Raises
        ------
        ValueError
            When convertion failed.
        """
        
        try:
            def get_full_text(elem):
                return (
                    "".join(elem.itertext()).strip() 
                    if elem is not None
                    else ""
                )
            
            vietdict_file = VIETDICT_DIR / "hoang_phe.xml"
            tree = ET.parse(vietdict_file)
            root = tree.getroot()
            
            support_attrib_map = {
                "ph": "phương ngữ",
                "vch": "văn chương",
                "trtr": "trang trọng",
                "kng": "khẩu ngữ",
                "thgt": "thông tục",
                "kc": "kiểu cách",
                "chm": "chuyên môn",
                "id": "ít dùng",
                "ng1": "nghĩa 1",
                "ng2": "nghĩa 2",
                "cũ": "cũ",
                "thường viết hoa": "thường viết hoa",
                "có trước": "có trước"
            }
            support_attrib_list = list(support_attrib_map.keys())

            words = []
            
            word_entries = root.findall("WORD")
            for word_entry in word_entries:
                word = get_full_text(word_entry.find("HEAD"))
                pos = get_full_text(word_entry.find("POS")).split(",")
                
                body_entries = word_entry.findall("BODY")
                for body_entry in body_entries:
                    meaning_entry = body_entry.find("MEANING")
                    support_attrib = None
                    meaning = None
                    
                    if meaning_entry is not None:
                        meaning = "".join(meaning_entry.itertext()).strip()
                        tags = re.findall(r"\[(.*?)\]", meaning)
                        
                        for tag in tags:
                            splits = [split.replace(",", "") for split in tag.split(" ")]
                            if splits[0] in support_attrib_list:
                                support_attrib = ",".join([
                                    support_attrib_map[split] for split in splits 
                                    if split in support_attrib_list
                                ])
                                meaning = meaning.replace(f"[{tag}] ", "")

                        if len(body_entries) > 1:
                            meaning = meaning[2:] 
                    
                    example = get_full_text(body_entry.find("EXAMPLE"))
                    reference = get_full_text(body_entry.find("REF"))
                    synonym = get_full_text(body_entry.find("SYN"))
                    antonym = get_full_text(body_entry.find("ANT"))

                    for p in pos:
                        words.append({
                            "Word": word,
                            "POS": p.strip(),
                            "Meaning": meaning,
                            "Example": example,
                            "Reference": reference,
                            "Synonym": synonym,
                            "Antonym": antonym,
                            "Support Attributes": support_attrib
                        })
            
            sorted_data = sorted(words, key=lambda x: x["POS"])
            out_csv = VIETDICT_DIR / "output" / "hoang_phe.csv"
            fieldnames = [
                "Word", 
                "POS",
                "Meaning",
                "Example",
                "Reference",
                "Synonym",
                "Antonym",
                "Support Attributes",
            ]
            
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_data)
                    
        except Exception as e:
            print(e)
            raise ValueError(f"Conversion failed for 'hoang_phe' dictionary")
    
    @staticmethod
    def __convert_wsd_frameworks():
        """
        Convert SemCor Training Corpora XML to CSV.

        Raises
        ------
        ValueError
            When convertion failed.
        """
        
        try:
            wsd_files = {
                "semcor": {
                    "data": EVAL_DIR / "Training_Corpora" / "SemCor" / "semcor.data.xml",
                    "gold": EVAL_DIR / "Training_Corpora" / "SemCor" / "semcor.gold.key.txt"
                },
                "semcor+omsti": {
                    "data": EVAL_DIR / "Training_Corpora" / "SemCor+OMSTI" / "semcor+omsti.data.xml",
                    "gold": EVAL_DIR / "Training_Corpora" / "SemCor+OMSTI" / "semcor+omsti.gold.key.txt"
                },
                "semeval2007": {
                    "data": EVAL_DIR / "Evaluation_Datasets" / "semeval2007" / "semeval2007.data.xml",
                    "gold": EVAL_DIR / "Evaluation_Datasets" / "semeval2007" / "semeval2007.gold.key.txt"
                },
                "semeval2013": {
                    "data": EVAL_DIR / "Evaluation_Datasets" / "semeval2013" / "semeval2013.data.xml",
                    "gold": EVAL_DIR / "Evaluation_Datasets" / "semeval2013" / "semeval2013.gold.key.txt"
                },
                "semeval2015":  {
                    "data": EVAL_DIR / "Evaluation_Datasets" / "semeval2015" / "semeval2015.data.xml",
                    "gold": EVAL_DIR / "Evaluation_Datasets" / "semeval2015" / "semeval2015.gold.key.txt"
                },
                "senseval2":  {
                    "data": EVAL_DIR / "Evaluation_Datasets" / "senseval2" / "senseval2.data.xml",
                    "gold": EVAL_DIR / "Evaluation_Datasets" / "senseval2" / "senseval2.gold.key.txt"
                },
                "senseval3":  {
                    "data": EVAL_DIR / "Evaluation_Datasets" / "senseval3" / "senseval3.data.xml",
                    "gold": EVAL_DIR / "Evaluation_Datasets" / "senseval3" / "senseval3.gold.key.txt"
                }
            }
            
            for key, value in wsd_files.items():
                sentences = []
            
                for _, sentence_entry in ET.iterparse(value["data"], events=("end",)):
                    if sentence_entry.tag == "sentence":        
                        sentence, lemma, pos = [], [], []
                        
                        sentence_id = sentence_entry.get("id")
                        for word_entry in sentence_entry:
                            word = word_entry.text
                            if len(word.split()) > 1:
                                word = word.replace(" ", "_")
                            
                            if word_entry.tag == "instance":
                                word = f"<{word}>"
                                
                            sentence.append(word)
                            lemma.append(
                                f"<{word_entry.get("lemma")}>" if word_entry.tag == "instance"
                                else word_entry.get("lemma")
                            )
                            pos.append(word_entry.get("pos"))
                            
                        sentences.append({
                            "SentenceId": sentence_id,
                            "Sentence": " ".join(sentence),
                            "Lemma": " ".join(lemma),
                            "POS": " ".join(pos)
                        })
                        
                        sentence_entry.clear()
                
                instances = []
                
                with open(value["gold"], mode="r", encoding="utf-8") as f:
                    for line in f:
                        ids = line.strip().split()
                        instance_id = ids[0]
                        sense_ids = ids[1:]
                        for sense_id in sense_ids:
                            lemma = wn.lemma_from_key(sense_id)
                            synset = lemma.synset()
                        
                            instances.append({
                                "InstanceId": instance_id,
                                "SenseId": sense_id,
                                "SynsetId": synset.name(),
                                "Definition": synset.definition()
                            })
                        
                sentence_fieldnames = [
                    "SentenceId", 
                    "Sentence", 
                    "Lemma", 
                    "POS"
                ]
                
                instance_fieldnames = [
                    "InstanceId",
                    "SenseId",
                    "SynsetId",
                    "Definition"
                ]
                
                output_dirs = WSD_DIR / "output" / key
                if not os.path.exists(output_dirs):
                    os.makedirs(output_dirs)
                
                sentence_csv = output_dirs / f"{key}_sentences.csv"
                instance_csv = output_dirs / f"{key}_instances.csv"
                with open(sentence_csv, "w", newline="", encoding="utf-8") as f_sentence, \
                    open(instance_csv, "w", newline="", encoding="utf-8") as f_instance:
                    
                    sentence_writer = csv.DictWriter(
                        f_sentence, 
                        fieldnames=sentence_fieldnames
                    )
                    sentence_writer.writeheader()
                    sentence_writer.writerows(sentences)
                    
                    instance_writer = csv.DictWriter(
                        f_instance, 
                        fieldnames=instance_fieldnames
                    )
                    instance_writer.writeheader()
                    instance_writer.writerows(instances)
                                                        
        except Exception as e:
            print(e)
            raise ValueError(f"Conversion failed for 'wsd' framework")
        
    @staticmethod
    def convert(object: str):
        objects = {
            "hoang_phe": XMLToCSVConverter.__convert_vietdict,
            "wsd_frameworks": XMLToCSVConverter.__convert_wsd_frameworks,
        }
    
        try:
            objects[object]()
            print(f"Conversion succeeded for '{object}'")
            
        except KeyError:
            raise ValueError(f"'{object}' is not available")
        
def main():
    try:
        pass
        # XMLToCSVConverter.convert("hoang_phe")
        XMLToCSVConverter.convert("wsd_frameworks")
    except ValueError as e:
        print(e)
    
if __name__ == "__main__":
    main()