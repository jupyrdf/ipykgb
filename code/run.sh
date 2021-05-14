#!/bin/bash
## Start Topic Name
startTopic="Star Wars"
## Number of Topics for Dataset
numOfTopics=100
## Dataset Filepath
dataset_path="./star_wars.txt"
## Cleaned Dataset Filepath
cleaned_dataset_path="./star_wars_cleaned.txt"
## Triples Dataset Filepath
triples_dataset_path="./star_wars_cleaned_triples.txt"
output_path="./new_star_wars.ttl"
kb_path="./starwars.ttl"
echo 'Starting Knowledge Graph Builder'
echo 'Running Dataset Creation'
python dataset_creation.py "$startTopic" $numOfTopics
echo 'Running Dataset Preprocessing'
python preprocessing.py "$dataset_path"
echo 'Running Entity Linking Trainer'
python entity_linking.py $cleaned_dataset_path
echo 'Running Triple Extraction'
python extract_triples.py $cleaned_dataset_path
echo 'Running Graph Expansion'
python graph_expand.py