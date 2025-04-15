import torch
import numpy as np
import pandas as pd
from datetime import datetime
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from torch.utils.data import Dataset, DataLoader

from .setup import default_tokenizer


# ------------------ WordNet ------------------ #

def simple_pos_tag(tokens):
    """
    A basic POS tagger that assigns tags based on rules.
    """
    tagged = []
    for token in tokens:
        tag = 'NN'
        if token.endswith('ly'):
            tag = 'RB'
        elif token.endswith('ing'):
            tag = 'VBG'
        elif token.endswith('ed'):
            tag = 'VBD'
        elif token.endswith('s') and not token.endswith('ss'):
            tag = 'NNS'

        common_verbs = [
            'is', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'go', 'goes', 'went', 'gone', 'see', 'saw', 'seen'
        ]
        if token.lower() in common_verbs:
            tag = 'VB'

        common_adj = [
            'good', 'bad', 'great', 'best', 'better', 'worst', 'worst', 'high',
            'low', 'big', 'small', 'large', 'little', 'many', 'much'
        ]
        if token.lower() in common_adj:
            tag = 'JJ'
        tagged.append((token, tag))
    return tagged


def enhance_text_with_wordnet(text):
    """
    Enhance text with WordNet features including synonyms and sentiment scores.
    Fully robust implementation that doesn't rely on NLTK's POS tagger.
    """
    try:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # simple robust tokenization
        simple_tokens = []
        for word in text.split():
            # handle common punctuation at the end of words
            if word and word[-1] in '.,:;!?)"\'':
                simple_tokens.append(word[:-1])
                simple_tokens.append(word[-1])
            # handle common punctuation at the start of words
            elif word and word[0] in '(\'\"':
                simple_tokens.append(word[0])
                simple_tokens.append(word[1:])
            else:
                simple_tokens.append(word)

        # remove empty tokens
        tokens = [token for token in simple_tokens if token]

        # use simple POS tagger instead of NLTK's
        try:
            tagged_tokens = simple_pos_tag(tokens)
        except Exception as pos_err:
            print(f"POS tagging error: {pos_err}")
            # fall back to just using tokens without enhancement
            return text

        # map simple POS tags to WordNet POS tags
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wn.ADJ
            elif tag.startswith('V'):
                return wn.VERB
            elif tag.startswith('N'):
                return wn.NOUN
            elif tag.startswith('R'):
                return wn.ADV
            else:
                return None

        enhanced_tokens = []
        for token, pos in tagged_tokens:
            # skip very short tokens, punctuations, and non-alphanumeric
            if len(token) <= 2 or not any(c.isalnum() for c in token):
                enhanced_tokens.append(token)
                continue

            # get WordNet POS
            wordnet_pos = get_wordnet_pos(pos)
            if not wordnet_pos:
                enhanced_tokens.append(token)
                continue

            # try to get synsets - robust error handling
            try:
                synsets = wn.synsets(token, pos=wordnet_pos)
            except Exception as wn_err:
                # just keep the original token if we can't get synsets
                enhanced_tokens.append(token)
                continue
            if not synsets:
                enhanced_tokens.append(token)
                continue

            # get the most common synset
            synset = synsets[0]

            # try to get SentiWordNet sentiment scores
            try:
                swn_synset = swn.senti_synset(synset.name())
                pos_score = swn_synset.pos_score()
                neg_score = swn_synset.neg_score()

                # add sentiment markers for strongly positive or negative words
                if pos_score > 0.6:
                    enhanced_tokens.append(f"{token}_POS")
                elif neg_score > 0.6:
                    enhanced_tokens.append(f"{token}_NEG")
                else:
                    enhanced_tokens.append(token)

                # add most common synonym for enrichment, but only for important words
                # to avoid too much noise
                if (wordnet_pos in [wn.NOUN, wn.VERB, wn.ADJ] and
                    len(synsets) > 1 and
                    len(synset.lemma_names()) > 1):
                    # avoid multi-word synonyms
                    synonyms = [
                        lemma for lemma in synset.lemma_names()
                        if lemma != token and '_' not in lemma
                    ]
                    if synonyms:
                        # take the first synonym and add it to the text
                        enhanced_tokens.append(synonyms[0])
            except Exception as swn_err:
                enhanced_tokens.append(token)

        return ' '.join(enhanced_tokens)
    except Exception as e:
        print(f"Error in WordNet enhancement: {e}")
        # return original text if enhancement fails
        return text if isinstance(text, str) else str(text)


# ------------------ Finalized Enhanced Dataset ------------------ #

class EnhancedDataset(Dataset):
    """
    The finalized Enhanced Dataset, which tokenize texts and create a WordNet dataset.
    """
    def __init__(
        self, texts, dates, labels, stock_names, tokenizer, max_chunk_length=256, max_chunks=8, use_wordnet=True
    ):
        self.texts = texts
        self.dates = dates
        self.labels = labels
        self.stock_names = stock_names
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length
        self.max_chunks = max_chunks
        self.use_wordnet = use_wordnet

        # encode stock names to indices
        self.stock_to_idx = {name: idx for idx, name in enumerate(sorted(set(stock_names)))}
        self.stock_indices = [self.stock_to_idx[name] for name in self.stock_names]

        # normalize time
        self.min_date = min(self.dates)
        self.time_values = [
            (pd.Timestamp(d).to_pydatetime() - pd.Timestamp(self.min_date).to_pydatetime()).days \
                for d in self.dates
        ]
        max_days = max(self.time_values) if self.time_values else 0
        self.time_values = \
            [days / max_days for days in self.time_values] if max_days > 0 else [0] * len(self.time_values)

        # enhance text
        self.enhanced_texts = [
            enhance_text_with_wordnet(str(t)) if use_wordnet else str(t) for t in self.texts
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.enhanced_texts[idx]
        time_value = self.time_values[idx]
        label = self.labels[idx]
        stock_idx = self.stock_indices[idx]

        words = text.split()
        chunks = [
            ' '.join(words[i:i+self.max_chunk_length//2]) \
                for i in range(0, len(words), self.max_chunk_length//2)
        ][:self.max_chunks]
        tokenized = [
            self.tokenizer(
                chunk, add_special_tokens=True, max_length=self.max_chunk_length,
                padding='max_length', truncation=True, return_tensors='pt'
            ) for chunk in chunks
        ]
        if not tokenized:
            tokenized = [
                self.tokenizer(
                    '', add_special_tokens=True, max_length=self.max_chunk_length,
                    padding='max_length', truncation=True, return_tensors='pt'
                )
            ]

        input_ids = torch.stack([t['input_ids'].squeeze(0) for t in tokenized])
        attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in tokenized])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'time_value': torch.tensor(time_value, dtype=torch.float),
            'stock_idx': torch.tensor(stock_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_temporal_batch(batch, label='label'):
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    time_values = torch.stack([item['time_value'] for item in batch])
    stock_indices = torch.stack([item['stock_idx'] for item in batch])
    labels = torch.stack([item[label] for item in batch])
    return input_ids_list, attention_mask_list, time_values, stock_indices, labels


# ------------------ Finalized Enhanced Dataset ------------------ #

def get_split_datasets(
    df, labels, train_size=0.7, val_size=0.15, tokenizer=default_tokenizer, use_wordnet=True,
    max_chunks=8, batch_size=4, max_chunk_length=256, model_label='bert_label', label='label',
    date='trading_day', post='post', stock='stock'
):
    # date time conversion and get the unique timestamps by interval
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values(date)
    time_interval_count = df[date].unique()
    train_idx = int(len(time_interval_count) * train_size)
    val_idx = int(len(time_interval_count) * (train_size + val_size))
    train_cutoff = time_interval_count[train_idx]
    val_cutoff = time_interval_count[val_idx]
    # get the label map that is more compatible to sentiment analysis models
    label_map = {label: i for i, label in enumerate(sorted(labels))}

    train_df = df[df[date] <= train_cutoff]
    val_df = df[(df[date] > train_cutoff) & (df[date] <= val_cutoff)]
    test_df = df[df[date] > val_cutoff]

    train_df[model_label] = train_df[label].map(label_map)
    val_df[model_label] = val_df[label].map(label_map)
    test_df[model_label] = test_df[label].map(label_map)

    train_texts = train_df[post].values
    train_dates = train_df[date].values
    train_labels = train_df[model_label].values
    train_stocks = train_df[stock].values

    val_texts = val_df[post].values
    val_dates = val_df[date].values
    val_labels = val_df[model_label].values
    val_stocks = val_df[stock].values

    test_texts = test_df[post].values
    test_dates = test_df[date].values
    test_labels = test_df[model_label].values
    test_stocks = test_df[stock].values

    print(f"Train set: {len(train_texts)} examples")
    print(f"Validation set: {len(val_texts)} examples")
    print(f"Test set: {len(test_texts)} examples")

    # WordNet enhancement test
    if use_wordnet:
        print("\nTesting WordNet enhancement...")
        for i in range(min(3, len(train_texts))):
            print(f"Original: {train_texts[i][:100]}...")
            enhanced = enhance_text_with_wordnet(str(train_texts[i]))
            print(f"Enhanced: {enhanced[:100]}...\n")

    train_dataset = EnhancedDataset(
        train_texts, train_dates, train_labels, train_stocks, tokenizer, max_chunk_length, max_chunks, use_wordnet
    )
    val_dataset = EnhancedDataset(
        val_texts, val_dates, val_labels, val_stocks, tokenizer, max_chunk_length, max_chunks, use_wordnet
    )
    test_dataset = EnhancedDataset(
        test_texts, test_dates, test_labels, test_stocks, tokenizer, max_chunk_length, max_chunks, use_wordnet
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_batch)
    return (
        (train_loader, train_df),
        (val_loader  , val_df  ),
        (test_loader , test_df ),
    )
