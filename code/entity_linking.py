import rdflib
import pandas as pd
import spacy
from spacy.kb import KnowledgeBase
import os
import csv
import random
import sys
from collections import Counter
from spacy.util import minibatch, compounding

nlp = spacy.load("en_core_web_lg")
graph = rdflib.Graph()
graph.parse('./starwars.ttl', format='turtle')


def text_format(text):
    text = text.lower().split(' ')

    resolved = ''
    for word in text:
        if len(word) > 2:
            resolved += word[0].upper() + word[1:]
        else:
            resolved += word
    return resolved


def get_description(entity, label):
    query_str = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?s ?p ?o
        WHERE {   
            ?s <https://swapi.co/vocabulary/desc> ?o.
            FILTER (?s=<%s>)
        }
    """ % label
    res = list(graph.query(query_str))
    description = str(res[0][2])
    return description


def get_label_results(text):
    query_str = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s ?o
        WHERE {
            ?s rdfs:label ?o.
            FILTER regex(?o, "%s")
        }
    """ % text
    res = list(graph.query(query_str))
    return res


def get_entities_to_train(dataset, num_of_entities=10):
    entities_to_train = {}
    descriptions = {}

    for index, row in dataset.iterrows():
        doc = nlp(dataset.iloc[index][0])
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON']:
                if ent.text not in entities_to_train:
                    res = get_label_results(ent.text)
                    for i in res:
                        desc = get_description(str(i[1]), str(i[0]))
                        if not desc == 'None':
                            entities_to_train[str(i[0])] = str(i[1])
                            descriptions[str(i[0])] = desc
                if len(entities_to_train) >= num_of_entities:
                    return entities_to_train, descriptions
    return entities_to_train, descriptions


def setup_spacy_KB(entities, descriptions):
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)
    for qid, desc in descriptions.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)

    # Add Aliases to spaCy KB
    for qid, name in entities.items():
        kb.add_alias(alias=text_format(name), entities=[qid], probabilities=[1])

    name_counts = {}
    for qid, name in entities.items():
        name = name.split()
        for i in name:
            if i not in name_counts:
                name_counts[i] = 1
            else:
                name_counts[i] += 1

    for name in name_counts:
        probs = [1 / name_counts[name] for i in range(name_counts[name])]
        qids = []
        for qid, qname in entities.items():
            if name in qname:
                qids.append(qid)
        kb.add_alias(alias=text_format(name), entities=qids, probabilities=probs)

    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"Aliases in the KB: {kb.get_alias_strings()}")

    for name in kb.get_alias_strings():
        print(f"Candidates for '{name}': {[c.entity_ for c in kb.get_candidates(name)]}")

    # Save KB
    output_dir = './entity_linking_data'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    kb.dump(output_dir + "/my_kb")

    return kb


def generate_training_data(dataset, entities_to_train, samples_threshold=10):
    training_data = {}

    for index, row in dataset.iterrows():
        sentence = dataset.iloc[index][0]
        for qid, name in entities_to_train.items():
            name = text_format(name)
            if qid in training_data and len(training_data[qid]) >= samples_threshold:
                continue

            # found = re.search("(^|[a-zA-Z\s])" + name + "($|[a-zA-Z\s])", sentence)
            doc = nlp(sentence)
            offset = None
            for ent in doc.ents:
                if name == str(ent):
                    offset = (ent.start_char, ent.end_char)
                    break

            # if found and offset is not None:
            if offset is not None:
                if qid not in training_data:
                    training_data[qid] = [sentence]
                else:
                    training_data[qid].append(sentence)

        print('*** Iteration {} of {} ***'.format(index, len(dataset)))
        completed = []
        for i, sentences in training_data.items():
            if len(sentences) == samples_threshold:
                completed.append(1)
            print('QID: {}, Total: {}'.format(i, len(sentences)))

        if len(completed) == len(entities_to_train.keys()):
            break

    return training_data


def format_dataset(training_data, entities_to_train):
    dataset = []
    for qid, sentences in training_data.items():
        for sentence in sentences:
            entity = text_format(entities_to_train[qid])
            doc = nlp(sentence)
            offset = None
            for ent in doc.ents:
                if entity == str(ent):
                    offset = (ent.start_char, ent.end_char)
                    break
            if offset is None:
                continue
            dataset.append((sentence, {"links": {offset: {qid: 1.0}}}))

    return dataset


def train(TRAIN_DOCS, train_iterations=500):
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
    with nlp.disable_pipes(*other_pipes):  # train only the entity_linker
        optimizer = nlp.begin_training()
        for itn in range(train_iterations):  # 500 iterations takes about a minute to train
            random.shuffle(TRAIN_DOCS)
            batches = minibatch(TRAIN_DOCS, size=compounding(4.0, 32.0, 1.001))  # increasing batch sizes
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.5,  # prevent overfitting
                    losses=losses,
                    sgd=optimizer,
                )
            if itn % 50 == 0:
                print(itn, "Losses", losses)  # print the training loss
    print(itn, "Losses", losses)


def evaluate_entity_linker(dataset, kb):
    aliases = list(kb.get_alias_strings())
    correct = 0
    total = 0
    for text, true_annot in dataset:
        # print(text)
        # print(f"Gold annotation: {true_annot}")
        doc = nlp(text)
        for ent in doc.ents:
            if str(ent) in aliases:
                if ent.kb_id_ == list(true_annot['links'][list(true_annot['links'].keys())[0]].keys())[0]:
                    correct += 1
                total += 1
    print('Correct: {} out of {}, Accuracy: {}'.format(correct, total, correct / total))


def training_pipeline(kb, dataset, entities, train_test_split=0.8):
    gold_ids = []
    for text, annot in dataset:
        for span, links_dict in annot["links"].items():
            for link, value in links_dict.items():
                if value:
                    gold_ids.append(link)
    print('Training Dataset Counts: ')
    print(Counter(gold_ids))

    train_dataset = []
    test_dataset = []

    samples_threshold = (len(dataset) / len(set(gold_ids)))
    train_range = int(samples_threshold * train_test_split)
    test_range = int(samples_threshold - train_range)

    for QID in entities.keys():
        indices = [i for i, j in enumerate(gold_ids) if j == QID]
        train_dataset.extend(dataset[index] for index in indices[0:train_range])
        test_dataset.extend(dataset[index] for index in indices[train_range:train_range+test_range])

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    TRAIN_DOCS = []
    for text, annotation in train_dataset:
        doc = nlp(text)
        TRAIN_DOCS.append((doc, annotation))

    entity_linker = nlp.create_pipe("entity_linker", config={"incl_prior": False})
    entity_linker.set_kb(kb)
    nlp.add_pipe(entity_linker, last=True)

    print('*** Starting Training Procedure ***')
    train(TRAIN_DOCS)
    print('*** Training Completed!')

    print('Evaluating with Training Dataset: ')
    evaluate_entity_linker(train_dataset, kb)

    print('Evaluating with Testing Dataset: ')
    evaluate_entity_linker(test_dataset, kb)


def main(dataset_path):
    dataset = pd.read_csv(dataset_path, delimiter='\n', header=None, error_bad_lines=False, quoting=csv.QUOTE_NONE)
    print('*** Retrieving Entities to Train ***')
    entities_to_train, descriptions = get_entities_to_train(dataset)
    with open('./entity_ids.txt', 'w') as file:
        file.write(str(entities_to_train))
    print('Training Entities: {}'.format(entities_to_train))

    print('*** Setting up spaCy Knowledge Base ***')
    kb = setup_spacy_KB(entities_to_train, descriptions)

    print('*** Generating Training Data ***')
    training_data = generate_training_data(dataset, entities_to_train)
    training_dataset = format_dataset(training_data, entities_to_train)
    training_pipeline(kb, training_dataset, entities_to_train)

    output_dir = './entity_linking_data'
    nlp.to_disk(output_dir + "/trained_el")
    print('*** Saved Custom Entity Linking Model! at {}'.format(output_dir))


def test_linker():
    output_dir = './entity_linking_data'
    nlp = spacy.load(output_dir + "/trained_el")
    text = 'Skywalker, also known as Darth Vader, is a fictional character in the Star Wars franchise'
    print('Original Text: {}'.format(text))
    # Preprocess
    doc = nlp(text)
    for np in doc.noun_chunks:
        noun = text_format(np.text)
        text = text.replace(np.text, noun)

    print('Preprocessed Text: {}'.format(text))
    # Predict
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)


if __name__ == '__main__':
    # if len(sys.argv) < 1:
    #     print('Usage: python preprocessing ./star_wars.txt')
    #     sys.exit()
    # dataset_path = str(sys.argv[1])
    # if "\r" in dataset_path:
    #     dataset_path = dataset_path.replace("\r", "")
    dataset_path = './star_wars_cleaned.txt'
    output_path = os.path.splitext(dataset_path)[0] + '_linked.txt'
    main(dataset_path)

    test_linker()
