__author__ = 'efrathaz'
import json
import re
from importlib import import_module
from Sentence import *
from Word import *


def init():
    create_stop_words()
    create_auxiliaries()
    upload_lex_models()
    

def create_stop_words():
    global stop_words
    f = open('static/stopwords.txt', 'r+')
    s = f.read()
    stop_words = re.split('\', \'|\", \"|\', \"|\", \'|[ \"|[ \'|\"]|\']', s)


def create_auxiliaries():
    global auxiliary_verbs
    f = open('static/auxiliaries.txt', 'r+')
    s = f.read()
    auxiliary_verbs = re.split('\', \'|\", \"|\', \"|\", \'|[ \"|[ \'|\"]|\']', s)
    

def upload_lex_models():
    global lex_model
    # TODO: add option for other models
    mod = import_module('lexDepModel')
    lex_model = mod.Embeddings.load('static/deps.words')

"""
def parse_unlabeled_sentences():
    sentences = []
    with open(app.config['UPLOAD_FOLDER']+'unlabeled.txt', 'r') as file:
        text = file.read()
        for sen in text.split("\n\n"):
            s = Sentence()
            s.depParse = sen
            text = ""
            for word in sen.split("\n"):
                fields = word.split()
                if len(fields) > 7:
                    s.add_word(Word(fields[0], fields[1], fields[2], fields[3], fields[6], fields[7]))
                    text += " " + fields[1]
            s.text = text
            sentences.append(s)
    return sentences
"""


def parse_unlabeled_sentences_text(text):
    sentences = []
    for sen in text.split("\n\n"):
        s = Sentence()
        s.depParse = sen
        t = ""
        for word in sen.split('\n'):
            fields = word.split()
            if len(fields) > 7:
                s.add_word(Word(fields[0], fields[1], fields[2], fields[3], fields[6], fields[7]))
                t += " " + fields[1]
        s.text = t
        sentences.append(s)
    return sentences


def parse_labeled_sentences_text(text):
    sentences = []
    for j in text.split('\n'):
        json_file = json.loads(j)
        s = Sentence()
        words = json_file['depParse'].split('\n')
        for word in words:
            fields = word.split()
            w = Word(fields[0], fields[1], fields[2], fields[3], fields[6], fields[7])
            s.add_word(w)
        update_roles(json_file, s)
        update_fee(json_file, s)
        s.luName = json_file['luName'].lower()
        s.luID = int(json_file['luID'])
        s.frameId = int(json_file['frameID'])
        s.text = json_file['text']
        sentences.append(s)
    return sentences

"""
def parse_labeled_sentences_zip():
    sentences = []
    archive = ZipFile(app.config['UPLOAD_FOLDER']+'labeled.zip', 'r')
    archive.extractall(app.config['UPLOAD_FOLDER']+'labeled')

    path = app.config['UPLOAD_FOLDER']+'labeled'
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            with open(item_path, 'r') as file:
                s = Sentence()
                json_file = json.load(file)
                words = json_file['depParse'].split('\n')
                for word in words:
                    fields = word.split()
                    w = Word(fields[0], fields[1], fields[2], fields[3], fields[6], fields[7])
                    s.add_word(w)
                update_roles(json_file, s)
                update_fee(json_file, s)
                s.luName = json_file['luName'].lower()
                s.luID = int(json_file['luID'])
                s.frameId = int(json_file['frameID'])
                s.text = json_file['text']
                sentences.append(s)
    return sentences
"""


def update_roles(json_file, sentence):
    for role in json_file['annotations']:
        indexes = list(x+1 for x in role['range'])
        w = find_root(indexes, sentence.words)
        w.set_role(role['name'], role['feID'])


def find_root(indexes, words):
    root = next((x for x in words if x.wordId == indexes[0]), None)
    for i in indexes:
        word = next((x for x in words if x.wordId == i), None)
        if word.head not in indexes:
            root = word
    if root.pos == 'in':
        indexes.remove(root.wordId)
        return find_root(indexes, words)
    return root


def update_fee(json_file, sentence):
    target = json_file['target']
    indexes = list(x+1 for x in target['range'])
    w = find_root(indexes, sentence.words)
    w.isFee = True
    sentence.fee = w


def get_target_predicates(sentence):
    """
    :param sentence: an unlabeled sentence
    :return: target predicates in sentence
    """
    targets = []
    for word in sentence.words:
        if (word.lemma not in stop_words) and check_pos(word.pos):
            targets.append(word)
    return targets


def check_pos(pos):
    """
    :param pos: part of speech
    :return: true if pos is noun, verb, adverb or adjective
    """
    if (pos[0] == 'v') or (pos[0] == 'n') or (pos[:2] == 'rb') or (pos[:2] == 'jj'):
        return True
    return False


def relevant_seeds(sentences, target, threshold):
    """
    :param sentences: a set of labeled sentences
    :param target: a target word of the unlabeled sentence
    :return: a set of relevant seed sentences
    """
    result = []
    for sentence in sentences:
        if lex(sentence.fee, target) > threshold:
            result.append(sentence)
    return result


def build_domain(sentence):
    """
    :param sentence
    :return: an alignment domain structure of sentence
    """
    words = list(sentence.words)
    domain = [sentence.fee]
    words.remove(sentence.fee)
    fee_id = sentence.fee.wordId

    # add words which directly depend on fee, except auxiliaries
    for word in words:
        if (word.head == fee_id) and (word.lemma not in auxiliary_verbs):
            domain.append(word)
            words.remove(word)
            # if word is a proposition or a conjunction node in the graph, add its direct dependents to domain
            if word.pos == 'in' or conjunction_node(sentence, word):
                for w in words:
                    if w.head == word.wordId:
                        domain.append(w)
                        words.remove(w)
    
    # add all words on complex paths from fee to any role-bearing words in the dependency graph
    paths = []
    for word in sentence.words:
        if word.roleName is not None:
            path_words, path = complex_path(sentence, word)
            for w in path_words:
                if w not in domain:
                    domain.append(w)
                    words.remove(w)
            paths.append(path)
    return domain, paths


def conjunction_node(sentence, word):
    """
    :param sentence:
    :param word:
    :return: True if word has 2 or more dependents in the dependency graph of sentence
    """
    dependents = 0
    for w in sentence.words:
        if w.head == word.wordId:
            dependents += 1
    if dependents > 1:
        return True
    return False


def complex_path(sentence, word):
    """
    :param sentence:
    :param word:
    :return: all the words in the complex path from fee to word: (nodes, edges)
    """
    words = [word]
    visited = []
    top = next((w.wordId for w in sentence.words if w.head == 0), None)  # root
    temp1 = sentence.fee
    temp2 = word

    # find lowest common ancestor
    while temp1.wordId not in visited and temp1.head != 0:
        visited.append(temp1.wordId)
        temp1 = sentence.words[temp1.head-1]  # temp1 = temp1.head

    while temp2.head != 0:
        if temp2.wordId in visited:
            top = temp2.wordId
            break
        visited.append(temp2.wordId)
        temp2 = sentence.words[temp2.head-1]  # temp2 = temp2.head

    temp1 = sentence.fee
    temp2 = word
    path1 = []
    path2 = []

    if temp1.wordId != top:
        path1.append(temp1.relation)
    if temp2.wordId != top:
        path2.append(temp2.relation)

    while temp1.wordId != top:
        temp1 = sentence.words[temp1.head-1]
        words.append(temp1)
        if temp1.wordId != top:
            path1.append(temp1.relation)

    while temp2.wordId != top:
        temp2 = sentence.words[temp2.head-1]
        if temp2 not in words:
            words.append(temp2)
            if temp2.wordId != top:
                path2.append(temp2.relation)

    path2.reverse()
    path1.extend(path2)
    return words, path1


def build_range(sentence, t, paths):
    """
    :param sentence
    :param t: a target word (FEE)
    :param paths: complex paths from FEE to all role bearing nodes in the corresponding alignment domain structure
    :return: an alignment range structure of sentence with t as target word
    """
    words = list(sentence.words)
    range_a = [t]
    words.remove(t)

    # add words which directly depend on t, except auxiliaries
    for word in words:
        if (word.head == t.wordId) and (word.lemma not in auxiliary_verbs):
            range_a.append(word)
            words.remove(word)
            # if word is a proposition or a conjunction node in the graph, add its direct dependents to range
            if word.pos == 'in' or conjunction_node(sentence, word):
                for w in words:
                    if w.head == word.wordId:
                        range_a.append(w)
                        words.remove(w)

    # add all words on complex paths from target to any possibly role-bearing words in the dependency graph
    # according to paths from the alignment domain
    for path in paths:
        to_add = words_in_path(sentence.words, path, t)
        if to_add is not None:
            for word in to_add:
                if word not in range_a:
                    range_a.append(word)
    return range_a


def words_in_path(words, path, first_word):
    """
    This function recursively checks if a complex path exists in a dependency tree of the sentence.
    If exists, returns the words in the path. Else returns None.
    :param words: a dependency tree (or subtree)
    :param path: a list of edges (semantic relations)
    :param first_word: the first word in the path
    :return: Words in the path, if exists, else None.
    """

    # BASE CASES:

    # we reached the end of the path, append empty list to the rest of the solution
    if not path:
        return []
    # there are not enough words to complete the path - the path doesn't exist
    if not words:
        return None

    # RECURSIVE PART:

    new_words = list(words)
    new_words.remove(first_word)

    # if the next node in the path is first_word's successor
    if first_word.relation == path[0]:
        head = next((x for x in words if x.wordId == first_word.head), None)
        if head is not None:
            rest = words_in_path(new_words, path[1:], head)
            if rest is not None:
                result = [head]
                result.extend(rest)
                return result

    # if the next node in the path is one of first_word's predecessors
    predecessors = list(x for x in words if x.head == first_word.wordId)
    for predecessor in predecessors:
        if predecessor.relation == path[0]:
            rest = words_in_path(new_words, path[1:], predecessor)
            if rest is not None:
                result = [predecessor]
                result.extend(rest)
                return result

    # else, the path doesn't exist
    return None


def complete(alignment, domain):
    """
    :param alignment
    :param domain: a predicate-argument structure
    :return: True if alignment covers all role-bearing nodes in domain, False otherwise
    """
    for word in domain:
        if word.roleName is not None:
            if next((pair[1] for pair in alignment if pair[0] == word), None) is None:
                return False
    return True


def assign(alignment, labeled, unlabeled):
    """
    :param alignment: an alignment
    :param labeled: a labeled sentence
    :param unlabeled: an unlabeled sentence
    :return: a labeled sentence according to the given alignment
    """

    # s = unlabeled
    s = Sentence()
    for word in unlabeled.words:
        s.add_word(Word(word.wordId, word.form, word.lemma, word.pos, word.head, word.relation))

    # update role bearing words
    for pair in alignment:
        domain_word = pair[0]
        range_word = pair[1]
        
        if range_word is not None:
            w = next((word for word in s.words if word.wordId == range_word.wordId), None)
            w.roleName = domain_word.roleName
            w.roleId = domain_word.roleId
            w.isFee = domain_word.isFee

            if w.isFee:
                s.fee = w

    # update sentence data
    s.depParse = unlabeled.depParse
    s.text = unlabeled.text
    start, end = find_word_indexes(s.text, s.fee)
    s.target = {'start': start, 'range': [(s.fee.wordId-1)], 'end': end, 'value': s.fee.form}
    s.frameId = labeled.frameId
    if s.fee.lemma == labeled.fee.lemma:
        s.luName = labeled.luName
        s.luID = labeled.luID
    annotations = []
    for word in s.words:
        if word.roleName is not None:
            start, end = find_word_indexes(s.text, word)
            annotations.append({'end': (end-1), 'name': word.roleName, 'value': word.form,
                                'start': start, 'range': [(word.wordId-1)], 'feID': word.roleId})
    s.annotations = annotations

    return s


def find_word_indexes(text, word):
    """
    :param text: a string representing a sentence
    :param word: a word
    :return: the indexes of the start and end of 'word' in 'text'
    """
    w_id = 1
    start = 0
    end = 0
    for w in text.split():
        if w_id == word.wordId:
            end = start + len(word.form)
            return start, end
        start += len(w) + 1
        w_id += 1
    return start, end


def highest_score_pairs(k, pair_list):
    """
    This function returns k sentences with the highest scores
    :param k: a number
    :param pair_list: a list of pairs (sentence, s) where sentence is a tagged sentence and s is its score
    """
    if k > len(pair_list):
        return pair_list
    result = []
    pair_list.sort(key=lambda x: x[1])
    for i in range(k):
        result.append(pair_list[i])
    return result


def find_optimal_alignment(a_domain, r):
    """
    :param a_domain: a predicate-argument structure of a labeled sentence
    :param r: a predicate-argument structure of an unlabeled sentence
    :return: an optimal alignment (list of word pairs) and its score
    """
    alpha = 0.55
    a_range = list(r)
    fee = a_domain[0]
    t = a_range[0]

    pa0 = [[fee, t]]  # partial alignment 0
    ps0 = lex(fee, t)  # partial alignment 0 score

    # initialize optimal alignment and score
    max_score = ps0
    max_alignment = list(pa0)
    for i in range(1, len(a_domain)):
        max_alignment.append([a_domain[i], None])

    # initialize stack
    stack = [[pa0, ps0]]

    # a = extension of pa0
    while len(stack) > 0:
        
        last = stack.pop()
        pa0 = last[0]
        ps0 = last[1]
        a = list(pa0)
        s = ps0
        k = len(a)

        # extend pa0 to a complete alignment 'a'
        for i in range(k, len(a_domain)):

            domain_word = a_domain[i]
            range_word = find_max_lex(domain_word, a_range)

            # if an alignment is found, add it to 'a' and add its score to 's'
            if range_word is not None:
                a.append([domain_word, range_word])
                s += lex(domain_word, range_word) + alpha*neighbours_num(a_domain, domain_word)
                a_range.remove(range_word)

            # if no alignment is found for domain_word, it's defined as unaligned
            else:
                a.append([domain_word, None])
        
        # if 'a' is maximal
        if s > max_score:

            if valid(a, k) and syn_maximizing(a, k):
                max_alignment = list(a)
                max_score = s

            else:

                domain_word_k = a_domain[k]

                # for range_word in (N - range(pa0)) + None
                for range_word in list((set(a_range).difference(set([pair[1] for pair in pa0]))).union({None})):
                    pa1 = list(pa0)
                    pa1.append([domain_word_k, range_word])
                    ps1 = ps0
                    ps1 += lex(domain_word_k, range_word)

                    for i in range(k):

                        domain_word_i = a_domain[i]
                        range_word_i = get_range_word(pa0, domain_word_i)

                        ps1 += (alpha * (syn(domain_word_k, domain_word_i, range_word, range_word_i)
                                + syn(domain_word_i, domain_word_k, range_word_i, range_word)))

                    stack.append([pa1, ps1])

    return max_alignment, max_score


def get_range_word(alignment, domain_word):
    """
    :param alignment:
    :param domain_word:
    :return: the corresponding word of domain_word in the given alignment
    """
    for pair in alignment:
        if pair[0] == domain_word:
            return pair[1]
    return None


def lex(w1, w2):
    """
    :param w1: a word
    :param w2: a word
    :return: the lexical similarity between w1 and w2 (value between 0 and 1)
    """
    if (w1 is None) or (w2 is None):
        return 0
    if w1.lemma == w2.lemma:
        return 1
    try:
        return lex_model.similarity(w1.lemma, w2.lemma)
    except KeyError:
        return 0


def syn(w1, w2, w3, w4):
    """
    :param w1: a word
    :param w2: a word
    :param w3: a word
    :param w4: a word
    :return: True if the semantic relation between w1 and w2 is equal to the semantic relation between w3 and w4
    """
    domain_relation = None
    range_relation = None
    
    if (w1 is None) or (w2 is None) or (w3 is None) or (w4 is None):
        return 0

    if w1.head == w2.wordId:
        domain_relation = w1.relation
    elif w2.head == w1.wordId:
        domain_relation = w2.relation
    if w3.head == w4.wordId:
        range_relation = w3.relation
    elif w4.head == w3.wordId:
        range_relation = w4.relation

    if (domain_relation is not None) and (domain_relation == range_relation):
        return 1

    return 0


def find_max_lex(w, words):
    """
    :param w: a word
    :param words: a set of words
    :return: a word in words with the maximal lexical similarity to w
    """
    max_word = None
    max_lex = 0

    for word in words:
        temp_lex = lex(w, word)
        if temp_lex > max_lex:
            max_lex = temp_lex
            max_word = word

    return max_word


def neighbours_num(structure, word):
    """
    :param structure: a predicate-argument structure
    :param word: a word
    :return: the number of neighbours of word in structure
    """
    neighbours = 1

    for w in structure:
        if w.head == word.wordId:
            neighbours += 1

    return neighbours


def valid(alignment, k):
    """
    :param alignment:
    :param k: a number
    :return: True if alignment is valid from the k-th item (i.e if alignment is an injective function)
    """
    range_words = []

    for pair in alignment[k:]:
        range_words.append(pair[1])

    # if a word appears in the alignment range more than once, then the alignment is not valid
    for word in range_words:
        if range_words.count(word) > 1 and word is not None:
            return False

    return True


def syn_maximizing(alignment, k):
    """
    This function checks for every edge e = (w1, w2) in alignment_domain and
    its correspondent edge e2 = (w3, w4) in alignment_range
    if they have the same label, i.e the same grammatical relation between their words.
    :param alignment: an alignment
    :param k: a number
    :return: True if for every edge e1 = (w1, w2) in alignment_domain and its correspondent
            edge e2 = (w3, w4) in alignment_range, r(e1) == r(e2), and False otherwise
    """

    partial_alignment = list(pair[0] for pair in alignment[k:])

    for w1 in list(pair[0] for pair in alignment):
        
        w2 = find_head(alignment, w1)
        
        if w2 is None:
            continue

        if (w1 in partial_alignment) or (w2 in partial_alignment):
            if syn(w1, w2, get_range_word(alignment, w1), get_range_word(alignment, w2)) != 1:
                return False

    return True


def find_head(alignment, word):
    for w in list(pair[0] for pair in alignment):
        if word.head == w.wordId:
            return w
    return None


def get_jason_data(s):
    data = {'depParse': s.depParse, 'target': s.target, 'text': s.text, 'luName': s.luName, 'luID': s.luID,
            'frameId': s.frameId, 'annotations': s.annotations}
    return data