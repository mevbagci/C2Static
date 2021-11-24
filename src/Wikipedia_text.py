import json
import os.path
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


set_articles = set()


def get_article_id_from_category_tree(tree_dir: str,searching_id: int, name_output: str, output_dir: str):
    global set_articles
    set_articles = set()
    set_count= set()
    with open(tree_dir, "r", encoding="UTF-8") as json_dir:
        tree_dict = json.load(json_dir)
        print(len(tree_dict))
        get_children(tree_dict, searching_id)
        for i in tree_dict:
            set_count.update(tree_dict[i]["articles"])
        print(len(set_count))
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{name_output}_{searching_id}.json", "w", encoding="UTF-8") as json_file:
        print(len(set_articles))
        list_set = list(set_articles)
        json.dump(list_set, json_file, indent=2, ensure_ascii=True)
        json_file.close()


visited_set = set()


def get_children(tree: dict, parent_id: int):
    global set_articles
    global visited_set
    if parent_id not in visited_set:
        visited_set.add(parent_id)
        if tree[f"{parent_id}"]["articles_count"] != 0:
            set_articles.update(set(tree[f"{parent_id}"]["articles"]))
            if tree[f"{parent_id}"]["children_count"] != 0:
                for child_id in tree[f"{parent_id}"]["children"]:
                    get_children(tree, child_id)


def combine_id_text(id_dir: str, text_to_id_dir: str, output_dir: str, outputname: str):
    counter = 0
    os.makedirs(output_dir, exist_ok=True)
    with open(id_dir, "r", encoding="UTF-8") as json_file:
        id_list = json.load(json_file)
    text_output = ""
    with open(text_to_id_dir, "r", encoding="UTF-8") as articles:
        for article in tqdm.tqdm(articles.readlines(), desc=f"Get all articles from {text_to_id_dir}"):
            info_article = article.split("\t")
            a_id = int(info_article[0])
            if a_id in id_list:
                counter += 1
                text = info_article[len(info_article)-1].replace("\n", "")
                text_output += f"{text}\n"
    with open(f"{output_dir}/{outputname}.txt", "w", encoding="UTF-8") as output_file:
        output_file.write(text_output)
        output_file.close()
    print(f"Number of saved articles {counter} from {len(id_list)}")


if __name__ == "__main__":
    dir_tree = "/mnt/rawindra/vol/public/baumartz/text2wiki/data/wiki/en/enwiki-20201120/enwiki-20201120-category-tree-without-all.json"
    search_id = 47397287
    dir_output = "/mnt/hydra/vol/public/bagci/C2Static/enwiki/20201120/articles"
    dir_output_text = f"/mnt/hydra/vol/public/bagci/C2Static/enwiki/20201120/articles/text_{search_id}"
    text_dir = "/mnt/hydra/vol/public/baumartz/wikipedia.v8/wiki_archive/enwiki/enwiki.token"
    name_output = "enwiki_20201120"
    print(f"Get all children from {search_id}")
    # get_article_id_from_category_tree(dir_tree, search_id, name_output, dir_output)
    combine_id_text(f"{dir_output}/{name_output}_{search_id}.json", text_dir, dir_output_text, f"{search_id}")
    lang = "de"
    # exit()
    # for lang in ["de"]:
    #     for speciality in ["Wirtschaft"]:
    #         # define parameter
    #         json_input_dir = f"/home/bagci/data/Wikipedia/Fachbuecher/{lang}/{speciality}/map_to_wiki/{lang}_economy_combined_register.json"
    #         wiki_file_input_dir = f"/home/bagci/data/Wikipedia/dewiki"
    #         wiki_file_name = f"wikipedia_{lang}.v8.token"
    #         dir_output = f"/home/bagci/data/Wikipedia/Fachbuecher/{lang}/{speciality}/wiki_text"
    #
    #         # Get all Words for language and specialitiy
    #         spec_wiki_id = get_all_article_id(json_input_dir)
    #
    #         # Get all text for selected articles and write them
    #         save_text_for_ids_and_combine(wiki_file_input_dir, wiki_file_name, dir_output, speciality, spec_wiki_id)
