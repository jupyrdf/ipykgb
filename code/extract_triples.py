import pandas as pd
import csv
import sys
from openie import StanfordOpenIE
import spacy
import os
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF , XSD, Namespace


def create_namespace(text, namespace='https://www.example.com/'):
    components = text.lower().split(' ')
    if len(components) > 1:
        new_text = components[0] + ''.join(x.title() for x in components[1:])
    else:
        new_text = text

    return Namespace(namespace) + new_text


def extract_relations(dataset, output_path, nlp):
    with StanfordOpenIE() as client:
        for index, row in dataset.iterrows():
            if index % 100 == 0:
                print('Processed {} of {}'.format(index, len(dataset)))

            text_to_id = {}
            text = dataset.iloc[index][0]
            doc = nlp(text)
            for ent in doc.ents:
                if ent.kb_id_ == 'NIL':
                    text_to_id[ent.text] = create_namespace(ent.text)
                else:
                    text_to_id[ent.text] = Namespace(ent.kb_id_)

            triples = []
            for triple in client.annotate(text):
                triples.append(triple)

            for t in triples:
                curr_triple = []
                for i in t:
                    if t[i] not in text_to_id:
                        id = create_namespace(t[i])
                    else:
                        id = text_to_id[t[i]]
                    curr_triple.append(id)
                with open(output_path, 'a', newline='') as file:
                    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                    wr.writerow(curr_triple)


def load_entity_linker():
    output_dir = './entity_linking_data'
    nlp = spacy.load(output_dir + "/trained_el")
    return nlp


def main(dataset_path, output_path):
    nlp = load_entity_linker()
    print('*** Extracting Triples ***')
    dataset = pd.read_csv(dataset_path, delimiter='\n', header=None, error_bad_lines=False, quoting=csv.QUOTE_NONE)
    extract_relations(dataset, output_path, nlp)
    print('*** DONE ***')


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: python extract_triples.txt ./star_wars_cleaned.txt')
        sys.exit()

    dataset_path = str(sys.argv[1])
    if "\r" in dataset_path:
        dataset_path = dataset_path.replace("\r", "")
    # dataset_path = 'star_wars_cleaned.txt'
    # output_path = 'star_wars_triples.txt'
    output_path = os.path.splitext(dataset_path)[0] + '_triples.txt'
    main(dataset_path, output_path)
