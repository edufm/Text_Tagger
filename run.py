from argparse import ArgumentParser
import search_engine as se

def main():
    parser = ArgumentParser()
    parser.add_argument('target_corpus', 
                        help='Nome do corpus NLTK alvo'
                        ,type= str, default= "reuters", nargs='?')
    
    parser.add_argument('query', 
                        help='Query a ser buscada',
                        type= str, default= "(asian (US rise)) (export minister)", nargs='?')
    
    parser.add_argument('DOSETUP', 
                        help='Se deve filtrar o corpus e montar repo e indexes',
                        type= bool, default= False, nargs='?')

    parser.add_argument('TESTMODE', 
                        help='Se deve rodar com o arquivo inteiro ou uma versão reduzida',
                        type= bool, default= True, nargs='?')

    args = parser.parse_args()
    print(args)









    corpus_file = (f"./storage/corpus_{args.target_corpus}_mini.json" if(args.TESTMODE) else f"./storage/corpus_{args.target_corpus}.json")
    repo_file   = f"./storage/repo_{args.target_corpus}.json"
    index_file  = f"./storage/index_{args.target_corpus}.json"
    semantic_index_file  = f"./storage/semantic_index_{args.target_corpus}.json"
    
    # "preset" ou "freq" 
    stopwords = (True, "freq")
    
    #"lower" ou "stemmer" 
    normalize = (True, "lower")
        
    query = args.query
    # ---------------------------- Verifica os inputs -------------------------
        # Verifia a query recebida
    if query.count("(") != query.count(")"):
        oppening = query.count("(")
        closing = query.count(")")
        raise ValueError(f"Unbalance number of parentesis in query, {oppening} - {closing}")
    
    # ---------------------------- Setup do Repo ----------------------------------
    if args.DOSETUP:
        # Cria o corpus
        se.gera_corpus.run(args.target_corpus, file=corpus_file, TESTMODE=args.TESTMODE)
        
        # Cria o repo e indice
        se.indexador.run(args.target_corpus, corpus_file, 
                         repo_file=repo_file, index_file=index_file, semantic_index_file=semantic_index_file,
                         stopwords=stopwords, normalize=normalize)
        
    # Lê o corpus
    corpus = se.repository.load(corpus_file)    

    # Lê o repo
    repo = se.repository.load(repo_file)

    # Lê o indice
    index = se.repository.load(index_file)
    
    # Lê o semantic_index
    semantic_index = se.repository.load(semantic_index_file)

    # -------------------------------- Querys -------------------------------------
    new_query, docids = list(se.search.busca_docids(index, query))
    
    # Calcula os diferentes score de cada doc
    tf_idf = se.search.rank(docids, index, repo)
    count = se.search.n_rank(new_query, docids, index, repo)
    semantic_query = se.search.semantic_rank(new_query, docids, index, repo)
    semantic = {docid:semantic_index[docid] for docid in docids}
    
    # Junta os scores num rank
    rank = {docid:tf_idf[docid] + count[docid] + semantic_query[docid] + 
            semantic[docid] for docid in docids}
    ranked = [k for k, v in sorted(rank.items(), key=lambda item: item[1])]

    print("Searhed for:", new_query)
    
    if len(ranked) > 0:
        print(ranked)
        print()
        print()
        for n, docid in enumerate(ranked):
            print("Result", n+1, ":")
            print(corpus[docid])
            print()
    else:
        print("No match for query")

if __name__ == '__main__':
    main()