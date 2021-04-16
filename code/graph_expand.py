import pandas as pd
import csv
import sys
import spacy
import os
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF, XSD, Namespace


def add_to_graph(dataset, graph):
    for index, row in dataset.iterrows():

        if index % 500 == 0:
            print('Added {} out of {} triples'.format(index, len(dataset)))

        triple = dataset.iloc[index]
        graph.add((URIRef(triple[0]), URIRef(triple[1]), URIRef(triple[2])))


def get_graph_length(graph):
    query_str = """
        SELECT ?s ?p ?o
        WHERE {   
            ?s ?p ?o.
        }
    """
    res = list(graph.query(query_str))
    return len(res)


def main(dataset_path, kb_path, output_path):
    print('*** Expanding {} Knowledge Graph'.format(kb_path))
    dataset = pd.read_csv(dataset_path, delimiter=',', header=None, error_bad_lines=False, quoting=csv.QUOTE_ALL)
    graph = Graph()
    graph.parse(kb_path, format='turtle')
    print('Graph Size: {} triples'.format(get_graph_length(graph)))
    print('*** Augmenting Graph ***')
    add_to_graph(dataset, graph)
    print('Final Graph Size: {} triples'.format(get_graph_length(graph)))

    with open(output_path, 'w') as file:
        file.write(graph.serialize(format='turtle').decode('utf-8'))


if __name__ == '__main__':
    # if len(sys.argv) < 1:
    #     print('Usage: python extract_triples.txt ./star_wars_cleaned.txt')
    #     sys.exit()
    #
    # dataset_path = str(sys.argv[1])
    # if "\r" in dataset_path:
    #     dataset_path = dataset_path.replace("\r", "")

    dataset_path = './star_wars_cleaned_triples.txt'
    output_path = './new_star_wars.ttl'
    kb_path = './starwars.ttl'
    main(dataset_path, kb_path, output_path)
