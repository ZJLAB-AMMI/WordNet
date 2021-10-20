# WordNet
## Motivation
[WordNet](https://wordnet.princeton.edu/) is an English word lexical knowledge base proposed by a research team from 
Princeton University. It explicitly depicts lexical relations between tens of thousands of word 
(as well as some phrases). We believe WordNet has great potential in data-driven natural language processing research. 
However, we found the original knowledge base that released by the WordNet research team is hard to parse.
Recently, some computational linguistic or NLP researchers, e.g., [ConvE](https://github.com/TimDettmers/ConvE), has 
tried to reorganize the knowledge base into an easy-to-parse format for tasks such as graph embedding. However, 
the file they released, for some unknown reasons, is incomplete. Therefore, based on the API from [NLTK](https://www.nltk.org/), 
we reorganized WordNet and released a complete, structured file.



## WordNet Introduction
![WordNet Data Structure](WordNet_Data_Structure.jpg)
The above figure generally describes the data structure of WordNet and some concept need to be explained.
- **Synset** (green circle): Synset is the core concept of WordNet. Synsets are the groupings of synonymous words (lemma) 
  that express the same concept. Each synset contains several lemmas, indicating the meaning of these lemmas are similar,
  and they are interchangeable in text. WordNet analyzes relations between four types of word, i.e., nouns (N), verbs 
  (v), adjectives (adj), and adverbs (adv). Of note, the name of a synset, e.g., car.n.01, can be regarded as a 
  mnemonic symbol. Changing the name of a synset won't influence the correctness of WordNet.
- **Lemma** (orange circle): Each lemma indicates an English word (or phrase in some occasions)
- **Glosses** (green rectangle): Each synset is affiliated a glosses, which is a sentence that describes the meaning of the synset.
  Some synsets also has extra example sentences to show how specific words can be used in sentences. However, we did not draw 
  example sentences in the figure as they are only available in a fraction of synsets.
- **Relations**
    - **Semantic Relations** (black arrow): Describes relationship between synsets. There are 22 semantic relations.
    - **Lexical Relations** (blue arrow): Describes relationship between synsets and lemmas. There are 3 lexical relations.
    - **Lemma Relations** (orange arrow): Describes relationship between lemmas. There are two lemma relations.
    
The Details of these relations can refer to **WordNet framework improvements for NLP Defining abstraction and 
scalability layers** (page 7-9) in this project.
****
### Relation Statistics
| Type  | Number | Type | Number| Type | Number
| :-----| :----  | :---- |:----|:---|:---|
|**Lexical Relations**||||
|Antonym| 7,979 |Derivationally Related Form|74,705|Pertainym|8,022|
|**Semantic Relations**|
|Hypernym|89,089|Hyponym|89,089|Instance Hypernym|8,577|
|Instance Hyponym|8,577|Member Holonym|12,293|Substance Holonym|797|
|Part Holonym|9,097|Member Memronym|12,293|Substance Memronym|797|
|Part Memronym|9,097|Topic Domain|6,654|In Topic Domain|6,654|
|Region Domain|1,360|In Region Domain|1,360|Usage Domain|1,376|
|In Usage Domain|1,376|Attribute|1,278|Entailment|408|
|Cause|220|Also See|3,272|Verb Group|1,750|
|Similar TOS|21,386|
|**Lemma Relations**|
|Lemma||206,941|Is Lemma|206,941|

## Data Structure of Parsed WordNet
    import pickle
    wordnet_obj = pickle.load(open('wordnet_KG.pkl'), 'rb')

The parsed WordNet is stored in wordnet_KG.pkl.
The data structure of this object is as:
- relation_idx_dict: maps relation name to an index
- idx_relation_dict: maps relation index to a name
- relation count: count relation numbers
- word_idx_dict: maps a synset or a lemma to an index. Note, each lemma is affiliated to a synset as each lemma has a
  synset prefix
- idx_word_dict: maps an index to a synset or a lemma
- synset definition: contains synset definition (glossary) information
- synset example dict: contains synset example sentences. Of note, not all synsets have example sentence
- fact_list: knowledge graph triplet (head, relation, tail) list (recorded in index format)
- unique_relation_set: knowledge graph triplet (head, relation, tail) set (recorded in string format)
- unique_normalized_word_set: all distinct word in wordnet (without synset prefix)
- lemma_full_name_set: all distinct lemma set
- normalized_index_dict: maps word to corresponding lemma (recorded in index)