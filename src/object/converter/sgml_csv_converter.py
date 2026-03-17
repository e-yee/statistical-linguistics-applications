import csv

from pathlib import Path
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[3]
EVBCORPUS_DIR = BASE_DIR / 'data' / 'evb_corpus'

class SGMLToCSVConverter:
    @staticmethod
    def __convert_evb_corpus():
        """
        Convert EVBCorpus SGML to CSV.

        Raises
        ------
        ValueError
            When convertion failed.
        """
        
        try:
            docs = []
            input_path =  EVBCORPUS_DIR / 'EVBCorpus_EVBNews_v2.0'
            
            for file in Path(input_path).glob('*.sgml'):
                with open(file, mode='r', encoding='utf-8') as f:
                    content = f.read()
                    
                soup = BeautifulSoup(content, 'html.parser')
                doc_id = soup.doc['id']
                spairs = soup.find_all('spair')

                for spair in spairs:
                    spair_id = spair.get('id')
                    
                    sentences = spair.find_all('s')
                    
                    en_sentence = sentences[0].text
                    vi_sentence = sentences[1].text
                    
                    docs.append({
                        'DocId': doc_id,
                        'SentencePairId': spair_id,
                        'EnglishSentence': en_sentence,
                        'VietnameseSentence': vi_sentence
                    })
            
            fieldnames = [
                'DocId', 
                'SentencePairId',
                'EnglishSentence',
                'VietnameseSentence'
            ]
            output_path = EVBCORPUS_DIR / 'output' / 'evb_corpus.csv'
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(docs)
                                                
        except Exception as e:
            print(e)
            raise ValueError(f'Conversion failed for "EVBCopus"')

    @staticmethod
    def convert(object: str):
        objects = {
            'EVBCorpus': SGMLToCSVConverter.__convert_evb_corpus,
        }
    
        try:
            objects[object]()
            print(f'Conversion succeeded for "{object}"')
            
        except KeyError:
            raise ValueError(f'"{object}" is not available')
        
def main():
    try:
        SGMLToCSVConverter.convert('EVBCorpus')
    except ValueError as e:
        print(e)
    
if __name__ == '__main__':
    main()