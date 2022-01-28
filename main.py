import json
import numpy as np
import pickle
import os

answers = np.array(json.load(open('answers.json', 'r')))
words = np.array(json.load(open('words.json', 'r')))

words_char = np.array([bytes(a, 'ascii') for a in words])
words_byte = words_char.view(('i1', 5)) - 97

answers_char = np.array([bytes(a, 'ascii') for a in answers])
answers_byte = answers_char.view(('i1', 5)) - 97

def first_occurance(a, v, axis=-1):
    return np.argmax(a == v, axis=axis)

def guess_responses(words_byte, candidates_byte):
    words_color = np.zeros(words_byte.shape[:-1]+(26,), 'i1')
    candidates_color = np.zeros(candidates_byte.shape[:-1]+(26,), 'i1')

    for w in range(words_byte.shape[0]):
        np.add.at(words_color[w], words_byte[w], 1)
    for c in range(candidates_byte.shape[0]):
        np.add.at(candidates_color[c], candidates_byte[c], 1)

    exists = np.minimum(words_color[:, np.newaxis], candidates_color[np.newaxis, :])

    responses = np.zeros((words_byte.shape[0], candidates_byte.shape[0], 5), 'i1')
    responses[words_byte[:, np.newaxis] == candidates_byte[np.newaxis, :]] = 2

    for w in range(words_byte.shape[0]):
        word_byte = words_byte[w]

        for i in range(word_byte.shape[0]):
            mask_done = responses[w, :, i] == 2
            exists[w, mask_done, word_byte[i]] -= 1

        for i in range(word_byte.shape[0]):
            mask_done = (exists[w, :, word_byte[i]] > 0) & (responses[w, :, i] == 0)
            responses[w, mask_done, i] = 1
            exists[w, mask_done, word_byte[i]] -= 1

    return responses

def word_to_byte(word):
    word_char = np.array(bytes(word, 'ascii'))
    word_byte = word_char.view(('i1', 5)) - 97
    return word_byte

def get_guess_scores(key):
    narr = np.apply_along_axis(np.bincount, -1, key + 128, minlength=256)
    scores = -np.sum(narr ** 2, axis=-1)
    return scores

def get_match_scores(key):
    score_from_key = key % 256
    score_from_key[score_from_key != 242] = 0
    return np.sum(score_from_key, axis=-1)

def number_to_key(number):
    responses_from_number = np.stack([number // 10**n % 10 for n in range(5)], axis=-1)
    return np.ravel_multi_index(np.rollaxis(responses_from_number, axis=-1).astype('i1'), (3,)*5).astype('i1')

def responses_to_key(responses):
    return np.ravel_multi_index(np.rollaxis(responses, axis=-1).astype('i1')[::-1], (3,)*5).astype('i1')

def key_to_responses(key):
    return np.moveaxis(np.array(np.unravel_index(key % 256, (3,)*5)).astype('i1')[::-1], 0, -1)

def responses_to_color(responses, word_byte):
    colors = np.zeros(responses.shape[:-1] + (26,), 'i1')
    colors[..., word_byte] = responses
    return colors

def hard_mask(word_prv, key_prv):
    word_prv_byte = word_to_byte(word_prv)
    color_prv = responses_to_color(key_to_responses(key_prv), word_prv_byte)
    responses_from_prv = guess_responses(np.array([word_prv_byte]), words_byte)[0]
    colors = responses_to_color(responses_from_prv, word_prv_byte)
    colors = np.array(colors)
    return np.all(color_prv <= colors, axis=-1)

try:
    keymap = pickle.load(open('keymap.pkl', 'rb'))
except FileNotFoundError:
    print("Creating a keymap...")
    responses = guess_responses(words_byte, answers_byte)
    keymap = responses_to_key(responses)
    pickle.dump(keymap, open('keymap.pkl', 'wb'), protocol=4)

hard_mode = True

if(hard_mode):
    print("Info: Hard mode is activated.")
print('=== Wordle solver ver. 1.0 ===')
print("1. Enter your guess\n"
      "2. Enter the color of each letter as number\n"
      "(e.g. 02100 corresponds to 'gray', 'green', 'yellow', 'gray', 'gray')")
mask = np.full(answers.shape[0], True)
hmask = np.full(words.shape[0], True)

niter = 1
key = 0

while niter <= 6 and np.sum(mask)>1:
    scores_guess = get_guess_scores(keymap[:, mask])
    scores_match = get_match_scores(keymap[:, mask])
    score = (scores_guess * 256 * 5 + scores_match)
    score = (score - np.min(score)) * hmask
    max_mask = score == np.max(score)

    print("\n=== Round #%d ===" % niter)
    print("Candidates: %d" % np.sum(mask))
    print("Suggestions: %s" % words[max_mask])
    while True:
        guess = input('Guess #%d: ' % niter)
        iw = np.searchsorted(words, guess)
        if words[iw] == guess:
            break
        else:
            print("Invalid word: %s" % guess)
    response_str = input('Response #%d: ' % niter)
    key = number_to_key(int(response_str))
    mask = mask & (keymap[np.searchsorted(words, guess)] == key)
    if hard_mode:
        hmask = hard_mask(guess, key)

    niter += 1

if np.any(mask):
    print("The answer is: %s" % answers[mask][0])
else:
    print("Error.")