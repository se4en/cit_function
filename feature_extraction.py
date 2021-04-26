from glob import iglob
import gzip
import os
import shutil
import json
import re
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

DATA_PATH = 'data/adjudicated-and-supplemental'
INFO_FILE = 'data/papers_info/arc-paper-ids.tsv'


# data loading

def match_id2files():
    """
    Match paper's ids with files' names
    """
    result = dict()

    # text files
    for text_file in iglob(DATA_PATH + "/*.xml"):
        file_name = os.path.basename(text_file)
        paper_id = file_name[:8]
        result[paper_id] = {
            "text" : file_name
        }
    # annotation files
    for ann_file in iglob(DATA_PATH + "/*.xml.ann"):
        file_name = os.path.basename(ann_file)
        paper_id = file_name[:8]
        if paper_id in result:
            result[paper_id]["ann"] = file_name
    # citation files
    for cit_file in iglob(DATA_PATH + "/*.xml.json"):
        file_name = os.path.basename(cit_file)
        paper_id = file_name[:8]
        if paper_id in result:
            result[paper_id]["cit"] = file_name

    return result


def get_annotation_info():
    """
    Extract annotation info for all papers
    """
    result = dict()

    for paper_id, files in id2files.items():
        paper_info = dict()
        with open(os.path.join(DATA_PATH, files["ann"]), 'r') as f:
            for l in f:
                ls = l.strip().split("\t")
                if ls[0].startswith("T"):
                    _type = ls[1].split(' ')[0].split('-')[0]
                    if _type == "Compares" or _type == "Contrasts":
                        _type = "Compares & Contrasts"

                    paper_info[ls[0]] = {
                        "type": _type,
                        "full_type": ls[1].split(' ')[0],
                        "strart": int(ls[1].split(' ')[1]),
                        "end": int(ls[1].split(' ')[2]),
                        "cit_name": ls[2]
                    }
        result[paper_id] = paper_info

    return result


def get_date_info():
    """
    Extract date info for all papers
    """
    with open(INFO_FILE, 'r') as f:
        df = pd.read_csv(f, sep='\t', header = None)
    return df.set_index(0)[1].to_dict()


# words frequency

def get_lemmas(text):
    """
    Compute list of lemmas from text
    """
    lemmatizer = WordNetLemmatizer()
    # only words
    words = list(filter(str.isalpha, text.split(' ')))
    # lower
    words = list(map(str.lower, words))
    # lemmatize
    words = list(map(lemmatizer.lemmatize, words))
    # del stop words
    words = [word for word in words if not word in stopwords.words()]
    return words


def get_unigram_info(train_papers):
    """
    Compute frequency for each word in all contexts
    """
    freq = dict()

    for paper_id in train_papers:
        with open(os.path.join(DATA_PATH, id2files[paper_id]["cit"])) as f:
            cit_data = json.load(f)

        if isinstance(cit_data['algorithms']['algorithm'], list):
            all_cits = cit_data['algorithms']['algorithm'][-1]['citationList']['citation']
        else:
            all_cits = cit_data['algorithms']['algorithm']['citationList']['citation']

        if not isinstance(all_cits, list):
            all_cits = [all_cits]

        for cits in all_cits:
            if 'contexts' not in cits:
                continue

            contexts = cits['contexts']['context']
            # if only 1 citation
            if not isinstance(contexts, list):
                contexts = [contexts]

            for cit in contexts:
                cit_id = cit['annotationId']
                words = get_lemmas(cit['#text'])
                for word in words:
                    if word in freq:
                        freq[word] += 1
                    else:
                        freq[word] = 1

    return freq


# data extraction for features

def get_authors(auth_list):
    """
    Prepare list of authors for comparing
    """
    all_names = set()

    for auth in auth_list:
        full_name = auth.split(' ', maxsplit=1)
        if len(full_name) > 1:
            if len(full_name[0]) > 1:
                all_names.add(full_name[0][0] + ' ' + full_name[1])
            all_names.add(' '.join(full_name))
        else:
            all_names.add(full_name[0])

    return all_names


def find_citations(context):
    """
    Find all citations in context
    """
    author = "(?:[A-Z][A-Za-z'`-]+)"
    etal = "(?:et al.?)"
    additional = "(?:,? (?:(?:and |& )?" + author + "|" + etal + "))"
    year_num = "(?:19|20)[0-9][0-9]"
    page_num = "(?:, p.? [0-9]+)?"  # Always optional
    year = "(?:, *"+year_num+page_num+"| *\("+year_num+page_num+"\))"
    regex = "(" + author + additional+"*" + year + ")"

    matches = re.findall(regex, context)
    return matches


def get_paper_info(paper_id):
    """
    Return count of words in paper, list of paper's authors
    and list of cited paper's authors
    """
    with open(os.path.join(DATA_PATH, id2files[paper_id]["text"])) as f:  # J00-3003
        data = f.read()
    paper = BeautifulSoup(data, "lxml")

    content_tag = paper.find("html").find("body").find("algorithms").find_all(recursive=False)
    citation_tag = content_tag[-1]
    content_tag = content_tag[0].find_all(recursive=False)[0]

    # cit_authors
    cit_authors = []
    for cit in citation_tag.find_all('citation'):
        for auth in cit.find_all('author'):
            cit_authors.append(auth.text)

    # words count
    words = re.split(';|,| |\n|\t', content_tag.text)
    words = [x for x in words if (x and x != ".")]
    words_count = len(words)

    # authors
    authors = []
    for author in content_tag.find_all("author"):
        authors.append(re.sub('[\n\t*.]', '', author.text))
    authors = get_authors(authors)

    return words_count, authors, cit_authors


# build features

def get_features(paper_id):
    """
    Compute features for paper
    """
    features = dict()
    words_count, paper_authors, authors_from_cit = get_paper_info(paper_id)
    if paper_id in date_info:
        paper_date = date_info[paper_id]
    else:
        paper_date = None

    with open(os.path.join(DATA_PATH, id2files[paper_id]["cit"])) as f:
        cit_data = json.load(f)

    if isinstance(cit_data['algorithms']['algorithm'], list):
        all_cits = cit_data['algorithms']['algorithm'][-1]['citationList']['citation']
    else:
        all_cits = cit_data['algorithms']['algorithm']['citationList']['citation']

    if not isinstance(all_cits, list):
        all_cits = [all_cits]

    for cits in all_cits:

        if 'authors' in cits:
            cit_authors = cits['authors']['author']
            # if only 1 author
            if not isinstance(cit_authors, list):
                cit_authors = [cit_authors]
            cit_authors = [auth['#text'] for auth in cit_authors]
            cit_authors = get_authors(cit_authors)  # set(cit_authors)
        else:
            cit_authors = set()

        if 'date' in cits and '#text' in cits['date'] and paper_date:
            cit_date = int(cits['date']['#text'])
            year_diff = paper_date - cit_date
        else:
            year_diff = None

        if 'contexts' not in cits:
            continue
        contexts = cits['contexts']['context']
        # if only 1 citation
        if not isinstance(contexts, list):
            contexts = [contexts]
        cit_cur_cnt = len(contexts)

        for cit in contexts:
            cit_id = cit['annotationId']
            if cit_id not in ann_info[paper_id]:
                continue

            cit_features = dict()

            # 2. citation density
            cit_features['citation_density'] = len(find_citations(cit['#text']))
            # 3. year difference
            cit_features['year_difference'] = year_diff
            # 4. citing location
            cit_position = int(cit['@endWordPosition'])
            cit_features['citing_location'] = cit_position / words_count
            # 5. citing frequency
            cit_features['citation_frequency'] = cit_cur_cnt
            # 6. number of Other citations with the same author
            other_cit_cnt = 0
            for auth in authors_from_cit:
                if auth in cit_authors:
                    other_cit_cnt += 1
            cit_features['other_citations '] = other_cit_cnt
            # 7. self reference
            cit_features['self_reference'] = bool(paper_authors.intersection(cit_authors))
            # 1. unigram features
            lemmas = get_lemmas(cit['#text'])
            for word in top_100:
                cit_features[word] = word in lemmas
            # target
            cit_features['class'] = ann_info[paper_id][cit_id]['type']

            features[cit_id] = cit_features
    return features


def extract_features(papers):
    """
    Compute features for all papers
    """
    result = dict()
    for paper_id in papers:
        for cit_id, features in get_features(paper_id).items():
            result[paper_id + '-' + cit_id] = features
    return result


if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('stopwords')

    id2files = match_id2files()
    ann_info = get_annotation_info()
    date_info = get_date_info()

    word_freq = get_unigram_info(list(ann_info.keys()))
    top_100 = sorted(word_freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top_100 = [pair[0] for pair in top_100[:100]]

    features = extract_features(list(ann_info.keys()))

    with open('features.pickle', 'wb') as f:
        pickle.dump(features, f)
