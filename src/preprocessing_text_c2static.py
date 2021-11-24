import spacy
import tqdm
from os import makedirs
from transformers import AutoTokenizer
from typing import List
from multiprocessing import Pool
from functools import partial


def spacy_txt_to_sentence(input_text: List[str], lang: str):
    """
    Part the text into sentences and save them
    :param lang: language of the spacy Model
    :param input_text: List of text
    """
    nlp = spacy.load(lang)
    all_stopwords = nlp.Defaults.stop_words
    text_output = ""
    for article in input_text:
        article = article.replace("\n", "")
        nlp.max_length = len(article) + 100
        doc = nlp(article, disable=["ner", "morphologizer"])
        for sentence in doc.sents:
            text_temp = []
            for token in sentence:
                text_temp.append(f"{token.lemma_}")
            text_output += f"{' '.join(text_temp)}\n"
    return text_output


def text_to_sentence(input_dir, lang, output_dir, model: str ="spacy"):
    with open(input_dir, "r", encoding="UTF-8") as text_file:
        text_lines = text_file.readlines()
        if model == "spacy":
            part_func = partial(spacy_txt_to_sentence, lang=lang)
            pool = Pool(6)
            result = list(tqdm.tqdm(pool.imap_unordered(part_func, text_lines),
                       desc=f"text_to_sentence with {model}"))
            pool.close()
            pool.join()
            with open(output_dir, "w", encoding="UTF-8") as output_write:
                for i in result:
                    output_write.write(i)
                output_write.close()
        text_file.close()


def txt_to_paragraph(input_dir: str, input_name: str, lang: str, paragraph_len: int, tokenizer: AutoTokenizer.from_pretrained):
    """
    Split the text into
    :param paragraph_len:
    :param tokenizer:
    :param input_dir:
    :param input_name:
    :param lang:
    :return:
    """
    nlp = spacy.load(lang)
    all_stopwords = nlp.Defaults.stop_words
    words = []
    bert_para = []
    makedirs(f"{input_dir}/paragraph", exist_ok=True)
    write_article = open(f"{input_dir}/paragraph/{input_name}", "w", encoding="UTF-8")
    counter = 0
    token_counter = 0
    with open(f"{input_dir}/{input_name}", "r", encoding="UTF-8") as text_file:
        for article in tqdm.tqdm(text_file, desc="Formate text in sentece for bert2static"):
            article = article.replace("\n", "")
            nlp.max_length = len(article) + 100
            doc = nlp(article, disable=["ner", "morphologizer"])
            for token in doc:
                token_counter += len(tokenizer.tokenize(token.lemma_))
                if token_counter >= paragraph_len:
                    write_article.write(f"{' '.join(words)}")
                    words = [token.lemma_]
                    token_counter = 0
                else:
                    words.append(token.lemma_)
        write_article.close()


if __name__ == "__main__":
    text_id = 25346631
    for language in ["en"]:
        model_name = "spacy"
        base_dir = f"/mnt/hydra/vol/public/bagci/C2Static/{language}wiki/20201120/articles/text_{text_id}"
        dir_text = f"{base_dir}/{text_id}.txt"
        out_dir = f"{base_dir}/{text_id}_{model_name}_sentence.txt"
        text_to_sentence(dir_text, language, out_dir, model_name)
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    #txt_to_paragraph("/home/bagci/data/Wikipedia/Fachbuecher/de/Wirtschaft/wiki_text", "Wirtschaft.txt", "de_core_news_sm", 512, tokenizer)
