import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from dsets import KnownsDataset, CounterFactDataset

import random
import shutil
import nltk
import numpy as np
import pickle
from tqdm import tqdm


def create_analogy_templates(all_analogies):
    """For each analogy, create a template using each word pair.

    Each template contains two special tokens: "[A]" and "[B]". These correspond
    to the first and second word of the analogies provided in the original
    analogies file, and will be substituted in during the rationalization phase.
    Each analogy also contains a distracor parenthetical clause, which doesn't
    contain any pertinent information for the analogies.
    """
    all_analogies['capital-common-countries']['template'] = (
        "When my flight landed in [B], I converted my currency and slowly fell "
        "asleep. (I had a terrifying dream about my grandmother, but that's a "
        "story for another time). I was staying in the capital, [A]")
    all_analogies['capital-world']['template'] = (
        "When my flight landed in [B], I converted my currency and slowly fell "
        "asleep. (I was behind on a couple of assignments, but I tried not to "
        "think about them). I was staying in the capital, [A]")
    all_analogies['currency']['template'] = (
        "As soon as I arrived in [A], I checked into my hotel and took a long "
        "nap. (I had finally finished the book I was reading and it was amazing). "
        "I had to figure out the exchange rate to the local currency, which is "
        "apparently called the [B]")
    all_analogies['city-in-state']['template'] = (
        "As soon as I arrived in [B], I checked into my hotel and watched a movie "
        "before falling asleep. (I had a great call with my husband, although I "
        "wish it were longer). I was staying in my favorite city, [A]")
    all_analogies['family']['template'] = (
        "I initially invited my [A], who gladly accepted my invitation. (My "
        "favorite song just came on, so I was able to relax). When I learned that "
        "women were allowed, I went ahead and also invited my [B]")
    all_analogies['gram1-adjective-to-adverb']['template'] = (
        "How could he do this so [B]? (I wasn't sure why my phone always rang at "
        "the most inopportune times). When I tried to do it, I could never be [A]")
    all_analogies['gram2-opposite']['template'] = (
        "I thought it was [A]. (Just then an ad came on the TV, but that's "
        "irrelevant). It was the opposite of that: it was [B]")
    all_analogies['gram3-comparative']['template'] = (
        "I knew it was [A], but that's before I saw it in person. (Just then I "
        "thought about my ex-wife, but I had to stop thinking about her). When I "
        "did end up seeing it in person, it was even [B]")
    all_analogies['gram4-superlative']['template'] = (
        "I thought it would be the [B] thing I'd ever encounter. (I tried to "
        "ignore my phone vibrating in my pocket). But when I did end up "
        "encountering it, it turned out it wasn't so [A]")
    all_analogies['gram5-present-participle']['template'] = (
        "Every other day, it started [B] in the morning. (I tried to remember the "
        "name of the woman at the bar). But today, it did not [A]")
    all_analogies['gram6-nationality-adjective']['template'] = (
        "I had never been friends with any [B] people before. (The funniest thing "
        "happened to me the other day, but that's a story for another time). In "
        "fact, I had never even been to [A]")
    all_analogies['gram7-past-tense']['template'] = (
        "Although I [B] yesterday, I had a million things to do today. (I "
        "suddenly felt a pinched nerve, so I made a mental note to get that "
        "checked out). So today I wouldn't have time to do any more [A]")
    all_analogies['gram8-plural']['template'] = (
        "I really wanted to buy the [A], more than I ever wanted to buy anything "
        "before. (I was also behind on my homework, but that's another story). So "
        "I went to the store and asked if they had any [B]")
    all_analogies['gram9-plural-verbs']['template'] = (
        "I can usually [A] by myself. (I was so behind on work but I tried to "
        "distract myself). Although it's so much better when someone else also "
        "[B]")
    return all_analogies


def preprocess_analogies(analogies):
    """Preprocess analogies by creating a dict containing all pairs.

    Args:
      analogies: A string containing the raw analogies file, provided by [1].

    Returns:
      all_analogies: A dict, where each key corresponds to a different analogy
        type (such as 'capital-common-countries'). Each value is a dict
        containing two keys: 'a' and 'b'. The value of these dicts are lists,
        both of which have the same length and contain the respective parts of
        each analogy. We only keep the analogies where both words are tokenized
        as single word-pieces using GPT-2's tokenizer (this is so metrics like
        antecedent percentage are well-defined, and not shared across multiple
        tokens).
    """
    all_analogies = {}
    split_analogies = " ".join(analogies).split(":")
    # There are 14 kinds of analogies. Go through each, and create a dict of
    # unique pairs that are both tokenized as single words.
    for analogy_index in range(1, len(split_analogies)):
        print('======== analogy_index', analogy_index)
        analogy_type = split_analogies[analogy_index].split(" ")[1]
        print('111 analogy_type', analogy_type)
        dense_analogies = split_analogies[analogy_index].split(" ")[2:]
        if analogy_index != len(split_analogies) - 1:
            dense_analogies = dense_analogies[:-1]  # Remove trailing whitespace.
        # `all_pairs` contains all word pairs, and includes repeats.
        all_pairs = [dense_analogies[0::2][i] + " " + dense_analogies[1::2][i]
                     for i in range(len(dense_analogies[0::2]))]
        unique_pairs = list(dict.fromkeys(all_pairs))  # this keeps them ordered
        first_parts = np.array([pair.split(" ")[0] for pair in unique_pairs])
        second_parts = np.array([pair.split(" ")[1] for pair in unique_pairs])

        all_analogies[analogy_type] = {}
        all_analogies[analogy_type]['a'] = first_parts
        all_analogies[analogy_type]['b'] = second_parts

    return all_analogies

def save(data, dir=None):
    with open(f'{dir}.pkl', 'wb') as file:
        pickle.dump(data, file)

def load(dir=None):
    with open(f'{dir}.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def get_predictions(mt, data, topk=10):
    results = []
    for d in tqdm(data):
        predictions = predict_token(
            mt,
            [d["prompt"]],
            return_p=True,
            topk=topk
        )
        results.append(predictions)

    return results

def match_tokens_with_scores(scores, ranges):

    test = []

    for b, e in ranges:
        word_score = torch.sum(scores[b:e])
        test.append(word_score)
    return torch.tensor(test)

def check_whitespace(prompt, tokens):
    results = []
    search_start = 0  # Track the current search position in the prompt

    for token in tokens:
        # breakpoint()
        # Find the starting index of the token from the current position
        start_index = prompt.find(token, search_start)

        has_whitespace_before = start_index > 0 and prompt[start_index - 1].isspace()

        if has_whitespace_before:
            token = " " + token
            results.append(token)
        else:
            results.append(token)

        # Update position to search for the next token
        search_start = search_start + len(token)

    return results




