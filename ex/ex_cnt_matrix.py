import MeCab
import numpy as np
import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import pickle

m = MeCab.Tagger()

target_tags = [
    'NNG',  # 일반 명사
    'NNP',  # 고유 명사
    'NNB',  # 의존 명사
    'NR',  # 수사
    'NP',  # 대명사
    'VV',  # 동사
    'VA',  # 형용사
    'MAG',  # 일반 부사
    'MAJ',  # 접속 부사
]


def parse_sentence(sentence, target_tags, stop_word):
    result = m.parse(sentence)
    temp = result.split('\n')
    temp_2 = [ sentence.split('\t') for sentence in temp]
    words = [ sentence[0] for sentence in temp_2 ]
    morphs = [ sentence[1].split(',')[0]
               for sentence in temp_2
               if len(sentence) > 1]
    morphs = [ morph for morph in morphs if morph in target_tags ]
    words = words[:len(morphs)]



    word_morph = [ (word,morph)
                   for morph, word in zip(morphs, words)
                   if word not in stop_word ]
    return word_morph


def extract_word_list(lyrics, target_tags, stop_word):
    result = []
    try:
        for idx in range(len(lyrics)):
            word_morph_list = parse_sentence(lyrics[idx], target_tags, stop_word)
            word = [ word_morph[0] for word_morph in word_morph_list if len(word_morph[0]) > 1]
            result.append(word)
    except:
        print(idx, '해당 인덱스에서 오류가 났습니다.')
    return result


df = pd.read_csv('../data/발라드.csv', encoding='cp949')
print(df.head())
lyrics = df['lyrics'].values
titles = df['title'].values
title_to_idx = { title:idx for idx, title in enumerate(titles) }

f = open("../data/ballad_title_to_idx.pkl", "wb")
pickle.dump(title_to_idx, f)
f.close()
print('pickle title_to_idx')
print(pickle.load(open("../data/ballad_title_to_idx.pkl" , 'rb')))



stop_word = ['것', '을', '겠', '은', '.', '는', ',']
word = extract_word_list(lyrics, target_tags, stop_word)

def make_bigram(word):
    return gensim.models.Phrases(word, min_count=5, threshold=100)

def make_trigram(word):
    bigram = gensim.models.Phrases(word, min_count=5, threshold=100)
    return gensim.models.Phrases(bigram[word], threshold=100)

#print(bigram)
def make_trigram_list(word, bigram_mod, trigram_mod):
    trigram_list = []
    for idx in range(len(word)):
        trigram_list.append(trigram_mod[bigram_mod[word[idx]]])
    return trigram_list

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = make_bigram(word)
trigram_mod = make_trigram(word)
trigram_list = make_trigram_list(word, bigram_mod, trigram_mod)
print(trigram_list[:10])
print(' '.join(trigram_list[0]))
lyrics = [' '.join(trigram) for trigram in trigram_list]
print(lyrics[0])


### TODO
# use CountVectorizer & TfidfVectorizer to construct dtm & dtm_tfidf
count = CountVectorizer(binary=False) # binary x, tf
tfidf = TfidfVectorizer()

dtm: np.ndarray = count.fit_transform(lyrics).toarray()
dtm_tfidf: np.ndarray = tfidf.fit_transform(lyrics).toarray()  # from csr_matrix to numpy array.  # 이제는 이거 한줄로 끝내기!
f = open("../classification_for_api/model/count_vectorizer.pkl", 'wb')
pickle.dump(count, f)
f.close()
print(pickle.load(open("../classification_for_api/model/count_vectorizer.pkl", 'rb')))

f = open("../data/dtm_tfidf.pkl", "wb")
pickle.dump(dtm_tfidf, f)
f.close()
print('pickle')
print(pickle.load(open("../data/dtm_tfidf.pkl" , 'rb')))
#print(np.argsort(dtm_tfidf[0]))
#print(list(tfidf.get_feature_names()))


#df = pd.DataFrame(data=dtm_tfidf, columns=list(tfidf.get_feature_names()))
#print(df.head())
##df.to_csv('../data/dtm_tfidf_matrix.csv')
#print('dtm.getfield()', count.get_feature_names())
#df = pd.DataFrame(data=dtm, columns=list(count.get_feature_names()))
#print(df.head())
#df.to_csv('../data/dtm_matrix_origin.csv', encoding='utf-8')
