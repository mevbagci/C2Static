import pickle
from datetime import datetime
import Wikipedia_text
import preprocessing_text_c2static
import make_vocab_dataset
import os
import learn_from_bert_ver2
import learn_from_bert_ver2_paragraph
from transformers import AutoTokenizer, BertTokenizer, BertModel

set_files = set()


def get_all_path_files(path_dir: str, end_file: str):
    global set_files
    for file in os.scandir(path_dir):
        if file.is_dir():
            get_all_path_files(file, end_file)
        elif (str(file.path)).endswith(f"{end_file}"):
            set_files.add(str(file.path))


if __name__ == "__main__":
    switch = {
        "ef":
            {
                "de": "de_core_news_sm",
                "en": "en_core_web_sm",
                "fr": "fr_core_news_sm",
                "da": "da_core_news_sm",
                "nl": "nl_core_news_sm",
                "el": "el_core_news_sm",
                "it": "it_core_news_sm",
                "zh": "zh_core_web_sm",
                "ja": "ja_core_news_sm",
                "lt": "lt_core_news_sm",
                "nb": "np_core_news_sm",
                "pl": "pl_core_news_sm",
                "pt": "pt_core_news_sm",
                "ro": "ro_core_news_sm",
                "ru": "ru_core_news_sm",
                "es": "es_core_news_sm",
                "multi": "xx_ent_wiki_sm",
                "default": "xx_ent_wiki_sm",
            },
        "ac":
            {
                "zh": "zh_core_web_trf",
                "da": "da_core_news_lg",
                "nl": "nl_core_news_lg",
                "en": "en_core_web_trf",
                "fr": "fr_dep_news_trf",
                "de": "de_dep_news_trf",
                "el": "el_core_news_lg",
                "it": "it_core_news_lg",
                "ja": "ja_core_news_lg",
                "lt": "lt_core_news_lg",
                "nb": "nb_core_news_lg",
                "pl": "pl_core_news_lg",
                "pt": "pt_core_news_lg",
                "ro": "ro_core_news_lg",
                "ru": "ru_core_news_lg",
                "es": "es_dep_news_trf",
                "multi": "xx_sent_ud_sm",
                "default": "xx_sent_ud_sm",
            }
    }
    bert_models = {
        "de": "bert-base-german-cased",
        "en": "bert-base-uncased"
    }
    # , "allenai/scibert_scivocab_uncased", "albert-base-v2"
    model_names = ["bert-base-uncased"]
    data_field_names = {
        "Economy": ["Economy_Quantitative-Finance_all_sentences.txt"],
        "Computer-Science": ["Computer-Science_all_sentences.txt"],
        "Wikipedia": ["wiki_per_sentence/wikipedia_all_per_sentences.txt"]
    }
    data_field_names_adapt = {
        "Economy": ["Economy_Quantitative-Finance_all_sentences_bigger_5_words.txt", "Economy_Quantitative-Finance_all_sentences_longer_140_chars.txt"],
        "Quantitative-Biology": ["Quantitative-Biology_all_sentences.txt"],
        "Wikipedia": ["wiki_per_sentence/wikipedia_all_per_sentences.txt"]
    }
    for model_name in model_names:
        for speciality in ["Wikipedia"]:
            # define parameter
            # json_input_dir = f"/home/bagci/data/Wikipedia/Fachbuecher/{lang}/{speciality}/map_to_wiki/{lang}_economy_combined_register.json"
            # wiki_file_input_dir = f"/home/bagci/data/Wikipedia/dewiki"
            # wiki_file_name = f"wikipedia_{lang}.v8.token"
            # dir_output = f"/home/bagci/data/Wikipedia/Fachbuecher/{lang}/{speciality}/wiki_text"
            # spacy_model = switch["ef"][f"{lang}"]
            base_dir = "/mnt/rawindra/vol/public/bagci/Arxiv/sum"
            out_base_dir = "/mnt/hydra/vol/public/bagci/project/Arxiv/sum"
            data_names = data_field_names[speciality]
            for data_name in data_names:
                input_dir = f"{base_dir}/{data_name}"
                # get_all_path_files(in_dir, ".txt")
                # input_dirs = set_files
                min_count = 10
                max_vocab_size = 750000
                num_epoch = 1
                batch_size = 256
                max_len = 512
                loss_print = 500
                embeddings_size = 768
                vocab_name = input_dir.split("/")[-1].replace(".txt", "")
                dir_output_model = out_base_dir.replace(f"/sum", f"/training/{speciality}/{model_name.replace('/','_')}")
                dir_output = out_base_dir.replace(f"/sum", f"/training/{speciality}/dataset/training_{vocab_name}")
                print(dir_output)
                run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

                # # Get all Words for language and specialitiy
                # spec_wiki_id = Wikipedia_text.get_all_article_id(json_input_dir)
                #
                # # Get all text for selected articles and write them
                # Wikipedia_text.save_text_for_ids_and_combine(wiki_file_input_dir, wiki_file_name, dir_output, speciality, spec_wiki_id)
                #
                # # Split in sentence
                # preprocessing_text_c2static.txt_to_sentence(f"{dir_output}", f"{speciality}.txt", spacy_model)
                #
                # # Split in paragraphs
                # preprocessing_text_c2static.txt_to_paragraph(f"{dir_output}", f"{speciality}.txt", spacy_model, 512, tokenizer)
                #
                # Convert into Dataset for static embeddings for sen
                # id2word, word2id, id2counts, word_counts = make_vocab_dataset.construct_vocab(f"{input_dir}", min_count, max_vocab_size)
                # os.makedirs(dir_output, exist_ok=True)
                # os.makedirs(dir_output_model, exist_ok=True)
                #
                # # creating dataset and vocab
                # pickle.dump(id2word, open(f"{dir_output}/id2word.p", "wb"))
                # pickle.dump(id2counts, open(f"{dir_output}/id2counts.p", "wb"))
                # pickle.dump(word_counts, open(f"{dir_output}/word_counts.p", "wb"))
                # pickle.dump(word2id, open(f"{dir_output}/word2id.p", "wb"))
                #
                # lines, words_locs, num_words = make_vocab_dataset.construct_dataset(f"{input_dir}", word2id)

                # Saving Dataset
                # with open(f"{dir_output}/dataset.p", "wb") as f:
                #     pickle.dump([lines, words_locs, num_words], f)

                devive_number = 1
                # BERT Model sentences
                os.system(f"python learn_from_bert_ver2.py --gpu_id {devive_number} --num_epochs {num_epoch} --algo SparseAdam --t 5e-6 --word_emb_size {embeddings_size} --location_dataset  "
                          f"{dir_output}  --model_folder {dir_output_model}  --batch_size {batch_size} --MAX_LEN {max_len} "
                          f"--num_negatives 10 --pretrained_bert_model {model_name} --print_loss_every {loss_print}")

                # os.system(f"python make_vocab_dataset.py --dataset_location {dir_output}/paragraph/{speciality}.txt --min_count {min_count} --max_vocab_size {max_vocab_size} --location_save_vocab_dataset "
                #           f"{dir_output}/paragraph/training_dataset/{run_name}/")

                # os.system(f"python learn_from_bert_ver2_paragraph.py --gpu_id 0 --num_epochs {num_epoch} --lr {lr} --algo SparseAdam --t 5e-6 --word_emb_size {embeddings_size} --location_dataset  "
                #           f"{dir_output}/paragraph/training_dataset/{run_name}/  --model_folder {dir_output}/paragraph/model/{run_name}  "
                #           f"--num_negatives 10 --pretrained_bert_model {bert_models[lang]}")
