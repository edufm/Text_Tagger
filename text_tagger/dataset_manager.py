from collections import defaultdict
import json

def save(data, path):
    data.to_csv(path)

def load(path):
    with open(path, 'r') as file:
        return json.load(file)


def create_index(repo):
    '''Indexa os documentos de um corpus.

    Args:
        repo: dicionario que mapeia docid para uma lista de tokens.

    Returns:
        O índice reverso do repositorio: um dicionario que mapeia token para
        lista de docids.
    '''
    indexed = defaultdict(lambda:defaultdict(int))
    
    for doc_id, words in repo.items():
        for word in words:
            indexed[word][doc_id] +=1

    return indexed