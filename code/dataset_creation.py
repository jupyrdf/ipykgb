import wikipedia
import spacy
import sys

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def generate_topics(curr_topic, topic_threshold):
    all_topics = []

    topic_count = 0
    index = 0
    while topic_count < topic_threshold:
        all_topics = list(set(all_topics))
        topics = wikipedia.search(curr_topic, suggestion=False)
        all_topics += topics
        curr_topic = all_topics[index]
        index += 1
        topic_count = len(all_topics)
        print('topic_count: {}'.format(topic_count))
    return all_topics


def generate_dataset(all_topics, dataset_name):
    total = 0
    for i, topic in enumerate(all_topics):
        try:
            content = wikipedia.WikipediaPage(topic).content
        except wikipedia.DisambiguationError:
            # This gets triggered when the topic is ambiguous for Wikipedia.
            continue

        doc = nlp(content)
        sentences = [sent.string.strip() for sent in doc.sents]
        with open('./' + dataset_name, 'a') as file:
            for sentence in sentences:
                file.write(sentence + '\n')
        total += len(sentences)
        print('Topic{}: {}, Sentences: {}, Total: {}'.format(i, topic, len(sentences), total))


def main(dataset_name, start_topic, num_of_topics):
    print('*** Generating Topics starting with {} ***'.format(start_topic))
    topics = generate_topics(start_topic, topic_threshold=int(num_of_topics))
    print('*** Creating Dataset ***')
    generate_dataset(topics, dataset_name)
    print('*** DONE ***')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python dataset_creation.py "Patrick Star" 100')
        sys.exit()
    start_topic = sys.argv[1]
    num_of_topics = sys.argv[2]
    dataset_name = start_topic.lower().replace(' ', '_').strip() + '.txt'
    main(dataset_name, start_topic, num_of_topics)
