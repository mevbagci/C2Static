import json
from os import makedirs
import tqdm


def get_all_article_id(json_dir: str) -> dict:
    all_article_dict = dict()
    with open(json_dir, "r", encoding="UTF-8") as json_file:
        all_register = json.load(json_file)
        for word_id in all_register:
            if word_id.isdigit():
                all_article_dict[word_id] = all_register[word_id]["name"]
    return all_article_dict


def save_text_for_ids_and_combine(wiki_dir: str, wiki_name: str, output_dir: str, special: str, article_id_dict: dict) -> str:
    text_combined = ""
    makedirs(output_dir, exist_ok=True)
    with open(f"{wiki_dir}/{wiki_name}", "r", encoding="UTF-8") as wiki_file:
        for article in tqdm.tqdm(wiki_file, desc=f"Get all text for speciality {special} language {lang}"):
            article_splitted = article.split("\t")
            article_id = article_splitted[0]
            article_text = article_splitted[2]
            if article_id in article_id_dict:
                text_combined += article_text
                with open(f"{output_dir}/{article_id}.txt", "w", encoding="UTF-8") as wiki_out:
                    wiki_out.write(article_text)
    with open(f"{output_dir}/{special}.txt", "w", encoding="UTF-8") as wiki_out:
        wiki_out.write(text_combined)
    return output_dir


if __name__ == "__main__":
    for lang in ["de"]:
        for speciality in ["Wirtschaft"]:
            # define parameter
            json_input_dir = f"/home/bagci/data/Wikipedia/Fachbuecher/{lang}/{speciality}/map_to_wiki/{lang}_economy_combined_register.json"
            wiki_file_input_dir = f"/home/bagci/data/Wikipedia/dewiki"
            wiki_file_name = f"wikipedia_{lang}.v8.token"
            dir_output = f"/home/bagci/data/Wikipedia/Fachbuecher/{lang}/{speciality}/wiki_text"

            # Get all Words for language and specialitiy
            spec_wiki_id = get_all_article_id(json_input_dir)

            # Get all text for selected articles and write them
            save_text_for_ids_and_combine(wiki_file_input_dir, wiki_file_name, dir_output, speciality, spec_wiki_id)


