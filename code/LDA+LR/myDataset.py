# nltk text processors
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset

import pandas as pd

from collections import Counter
from functools import partial
import re # regular expression

from tqdm import tqdm, tqdm_notebook # show progress bar

from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import  gensim
def build_vocab(corpus):
    vocab = {}
    for doc in corpus:
        for token in doc:
            if token not in vocab.keys():
                vocab[token] = len(vocab)
    return vocab

def build_index2token(vocab):
    index2token = {}
    for token in vocab.keys():
        index2token[vocab[token]] = token
    return index2token

def tokenize(text, stop_words, lemmatizer):
    text = re.sub(r'[^\w\s]', '', text) # remove special characters
    text = text.lower() # lowercase
    tokens = wordpunct_tokenize(text) # tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens] # noun lemmatizer  词性还原
    tokens = [lemmatizer.lemmatize(token, "v") for token in tokens] # verb lemmatizer
    tokens = [token for token in tokens if token not in stop_words] # remove stopwords
    return tokens

def build_bow_vector(sequence, idx2token):
    vector = [0] * len(idx2token)
    for token_idx in sequence:
        if token_idx not in idx2token:
            raise ValueError('Wrong sequence index found!')
        else:
            vector[token_idx] += 1
    return vector
def replace_numbers(tokens):
    return [re.sub(r'[0-9]+', '<NUM>', token) for token in tokens]

def remove_rare_words(tokens, common_tokens, max_len):
    return [token if token in common_tokens else '<UNK>' for token in tokens][-max_len:]


class myDataset(Dataset):
    # def __init__(self, data_path, max_vocab=5000, max_len=128):
    def __init__(self, data_path, max_vocab=200000, max_len=180000):
        df = pd.read_csv(data_path)
        status_dict = df['label'].unique().tolist()
        df['label'] = df['label'].apply(lambda x: status_dict.index(x))
        # Clean and tokenize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        df['tokens'] = df.content.apply(
            partial(
                tokenize,
                stop_words=stop_words,
                lemmatizer=lemmatizer,
            ),
        )

        all_tokens = [token for doc in list(df.tokens) for token in doc]

        # Build most common tokens bound by max vocab size
        common_tokens = set(
            list(
                zip(*Counter(all_tokens).most_common(max_vocab))
            )[0]
        )

        # Replace rare words with <UNK>
        tqdm.pandas(desc="Replace rare words with <UNK>")
        df.loc[:, 'tokens'] = df.tokens.progress_apply(
            partial(
                remove_rare_words,
                common_tokens=common_tokens,
                max_len=max_len,
            ),
        )

        # Replace numbers with <NUM>

        tqdm.pandas(desc="Replace numbers with <NUM>")
        df.loc[:, 'tokens'] = df.tokens.progress_apply(replace_numbers)

        # Remove sequences with only <UNK>
        tqdm.pandas(desc="Remove sequences with only <UNK>")
        df = df[df.tokens.progress_apply(
            lambda tokens: any(token != '<UNK>' for token in tokens),
        )]

        # Build vocab
        vocab = sorted(set(
            token for doc in list(df.tokens) for token in doc
        ))
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        # Convert tokens to indexes
        tqdm.pandas(desc="Convert tokens to indexes")
        df['indexed_tokens'] = df.tokens.progress_apply(
            lambda doc: [self.token2idx[token] for token in doc],
        )

        # Build BoW vector
        tqdm.pandas(desc=" Build BoW vector")
        df['bow_vector'] = df.indexed_tokens.progress_apply(

            build_bow_vector, args=(self.idx2token,)

        )

        # Build TF-IDF vector
        vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=lambda doc: doc,
            preprocessor=lambda doc: doc,
            token_pattern=None,
        )
        vectors = vectorizer.fit_transform(df.tokens).toarray()
        df['tfidf_vector'] = [vector.tolist() for vector in vectors]
        # build LDA vector
        common_texts = df.tokens

        # 把文章转换成list
        dictionary = Dictionary(common_texts)
        # 把文本转换成词袋的形式  id：freq
        corpus = [dictionary.doc2bow(text) for text in common_texts]
        lda = LdaModel(corpus, id2word=dictionary, num_topics=7, iterations = 600,chunksize = 80,passes = 60)
        corpus_topic = lda[corpus]
        numpy_matrix = gensim.matutils.corpus2dense(corpus_topic, num_terms=7)
        numpy_matrix = numpy_matrix.T
        df['LDA_vector']=[vector.tolist() for vector in numpy_matrix]


        self.LDA_vector=df.LDA_vector.tolist()
        self.text = df.content.tolist()
        self.sequences = df.indexed_tokens.tolist()
        self.bow_vector = df.bow_vector.tolist()
        self.tfidf_vector = df.tfidf_vector.tolist()
        self.targets = df.label.tolist()
        self.df=df

    def __getitem__(self, i):

        return (
            self.sequences[i],
            self.bow_vector[i],
            self.tfidf_vector[i],
            self.targets[i],
            self.text[i],
            self.LDA_vector[i],
        )

    def __len__(self):
        return len(self.targets)
