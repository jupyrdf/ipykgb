import sys
sys.path.append('../../')
import pandas as pd
import spacy
import neuralcoref
import os
import csv

nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def text_format(text):
    text = text.lower().split(' ')

    resolved = ''
    for word in text:
        if len(word) > 2:
            resolved += word[0].upper() + word[1:]
        else:
            resolved += word
    return resolved


def add_to_dataset(resolved_text, dataset_index, keywords, output_path):
    doc = nlp(resolved_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    for sentence in sentences:
        doc = nlp(sentence)
        # add to keywords
        for np in doc.noun_chunks:
            noun = text_format(np.text)
            if noun not in keywords:
                keywords[noun] = 1
            else:
                keywords[noun] += 1
            # replace keyword in sentence
            sentence = sentence.replace(np.text, noun)
        if len(sentence) < 10:
            continue
        with open(output_path, "a") as file:
            file.write(sentence + "\n")
        dataset_index += 1
    return dataset_index


def process_dataset(dataset, output_path, coref_batch_size=3):
    text = ''
    dataset_index = 0
    keywords = {}
    for index, row in dataset.iterrows():
        text += ' ' + dataset.iloc[index][0]
        if index == 0:
            continue
        if index % coref_batch_size == 0:
            doc = nlp(text)
            resolved_text = doc._.coref_resolved
            dataset_index = add_to_dataset(resolved_text, dataset_index, keywords, output_path)
            text = ''

        if index % 100 == 0:
            print('Completed Batch: {}'.format(index))


def main(dataset_path, output_path):
    dataset = pd.read_csv(dataset_path, delimiter='\n', header=None, error_bad_lines=False, quoting=csv.QUOTE_NONE)
    print('*** Preprocessing Dataset ***')
    process_dataset(dataset, output_path)
    print('*** DONE ***')


if __name__ == '__main__':
    # if len(sys.argv) < 1:
    #     print('Usage: python preprocessing ./star_wars.txt')
    #     sys.exit()
    # dataset_path = str(sys.argv[1])
    # if "\r" in dataset_path:
    #     dataset_path = dataset_path.replace("\r", "")
    dataset_path = './star_wars.txt'
    output_path = os.path.splitext(dataset_path)[0] + '_cleaned.txt'
    main(dataset_path, output_path)
