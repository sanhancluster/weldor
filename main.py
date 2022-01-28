import json
import numpy as np
import pickle

answers = np.array(json.load(open('answers.json', 'r')))
words = np.array(json.load(open('words.json', 'r')))

words_char = np.array([bytes(a, 'ascii') for a in words])
words_byte = words_char.view(('i1', 5)) - 97

answers_char = np.array([bytes(a, 'ascii') for a in answers])
answers_byte = answers_char.view(('i1', 5)) - 97


def guess_responses(words_byte, candidates_byte):
    words_color = np.zeros(words_byte.shape[:-1] + (26,), 'i1')
    candidates_color = np.zeros(candidates_byte.shape[:-1] + (26,), 'i1')

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


def get_guess_scores(answer_mask_now):
    key_now = keymap[:, answer_mask_now]
    n_arr = np.apply_along_axis(np.bincount, -1, key_now % 256, minlength=243)
    scores = -np.sum(n_arr ** 2, axis=-1) + answers.shape[0] ** 2
    return scores


def get_match_scores(answer_mask_now):
    key_now = keymap[:, answer_mask_now]
    score_from_key = key_now % 256
    score_from_key[score_from_key != 242] = 0
    return np.sum(score_from_key, axis=-1)


def number_to_key(number):
    responses_from_number = np.stack([number // 10 ** n % 10 for n in range(5)], axis=-1)
    return np.ravel_multi_index(np.rollaxis(responses_from_number, axis=-1).astype('i1'), (3,) * 5).astype('i1')


def responses_to_key(responses):
    return np.ravel_multi_index(np.rollaxis(responses, axis=-1).astype('i1')[::-1], (3,) * 5).astype('i1')


def key_to_responses(key):
    return np.moveaxis(np.array(np.unravel_index(key % 256, (3,) * 5)).astype('i1')[::-1], 0, -1)


def responses_to_color(responses, word_byte):
    colors = np.zeros(responses.shape[:-1] + (26,), 'i1')
    colors[..., word_byte] = responses
    return colors


def hard_mask(word_prv, key_prv, word_mask_now):
    word_prv_byte = word_to_byte(word_prv)
    color_prv = responses_to_color(key_to_responses(key_prv), word_prv_byte)
    responses_from_prv = guess_responses(np.array([word_prv_byte]), words_byte[word_mask_now])[0]
    colors = responses_to_color(responses_from_prv, word_prv_byte)
    colors = np.array(colors)
    word_mask_now[word_mask_now] = np.all(color_prv <= colors, axis=-1)


def get_score(word_mask_now, answer_mask_now):
    scores_guess = get_guess_scores(answer_mask_now)
    scores_match = get_match_scores(answer_mask_now)
    score_now = (scores_guess * 256 * 5 + scores_match) * word_mask_now
    return score_now


import tqdm
from joblib import delayed, Parallel

def try_words(n_search, word_mask_now, answer_mask_now):
    score = get_score(word_mask_now, answer_mask_now)
    n_search = np.minimum(n_search, np.sum(word_mask_now))
    trials = np.argsort(score)[::-1][:n_search]

    def try_word(t):
        probability = np.bincount(keymap[t, answer_mask_now] % 256, minlength=243)
        score_trial = 0
        for key_trial in np.arange(243).astype('i1')[probability > 0]:
            answer_mask_trial = answer_mask_now & (keymap[t] == key_trial)
            score_trial += probability[key_trial] * np.max(get_guess_scores(answer_mask_trial))
        return score_trial

    scores_trial = Parallel(n_jobs=4)(delayed(try_word)(t) for t in trials)
    return trials[np.argsort(scores_trial)[::-1]]


def round(niter, n_candidates, suggestions):
    print("\n=== Round #%d ===" % niter)
    print("Candidates: %d" % n_candidates)
    print("Suggestions: %s" % suggestions)
    while True:
        guess = input('Guess #%d: ' % niter)
        iw = np.searchsorted(words, guess)
        if words[iw] == guess:
            break
        else:
            print("Invalid word: %s" % guess)
    response_str = input('Response #%d: ' % niter)
    key = number_to_key(int(response_str))
    return guess, key


try:
    keymap = pickle.load(open('keymap.pkl', 'rb'))
except FileNotFoundError:
    print("Creating a keymap...")
    responses = guess_responses(words_byte, answers_byte)
    keymap = responses_to_key(responses)
    pickle.dump(keymap, open('keymap.pkl', 'wb'), protocol=4)

hard_mode = True
n_search = 64

if __name__ == "__main__":
    if hard_mode:
        print("Info: Hard mode is activated.")
    print('=== Wordle solver ver. 1.0 ===')
    print("1. Enter your guess\n"
          "2. Enter the color of each letter as number\n"
          "(e.g. 02100 corresponds to 'gray', 'green', 'yellow', 'gray', 'gray')")
    answer_mask = np.full(answers.shape[0], True)
    word_mask = np.full(words.shape[0], True)

    niter = 0
    key = 0

    while niter <= 6 and np.sum(answer_mask) > 1:
        niter += 1
        score = get_score(word_mask, answer_mask)
        trials = try_words(n_search, word_mask, answer_mask)

        suggestions = words[trials]
        #suggestions = words[score == np.max(score)]
        guess, key = round(niter, np.sum(answer_mask), suggestions)

        w = np.searchsorted(words, guess)
        answer_mask = answer_mask & (keymap[w] == key)
        if hard_mode:
            word_mask = hard_mask(guess, key, word_mask)

    if np.any(answer_mask):
        print("The answer is: %s" % answers[answer_mask][0])
    else:
        print("Error.")
