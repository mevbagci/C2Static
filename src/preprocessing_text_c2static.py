import spacy
import tqdm
from os import makedirs


def txt_to_sentence(input_dir: str, input_name: str, lang: str):
    """
    Part the text into sentences and save them
    :param input_name: name of the input
    :param lang: language of the spacy Model
    :param input_dir: input for txt data
    """
    nlp = spacy.load(lang)
    all_stopwords = nlp.Defaults.stop_words
    makedirs(f"{input_dir}/sentences", exist_ok=True)
    write_article = open(f"{input_dir}/sentences/{input_name}", "w", encoding="UTF-8")
    with open(f"{input_dir}/{input_name}", "r", encoding="UTF-8") as text_file:
        for article in tqdm.tqdm(text_file, desc="Formate text in sentece for bert2static"):
            article = article.replace("\n", "")
            nlp.max_length = len(article) + 100
            doc = nlp(article, disable=["ner", "morphologizer"])
            for sentence in doc.sents:
                text_temp = []
                for token in sentence:
                    text_temp.append(f"{token.lemma_}")
                write_article.write(f"{' '.join(text_temp)}\n")
    write_article.close()


if __name__ == "__main__":
    txt_to_sentence("/home/bagci/data/Wikipedia/Fachbuecher/de/Wirtschaft/wiki_text", "Wirtschaft.txt", "de_core_news_sm")
