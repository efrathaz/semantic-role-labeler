__author__ = 'efrathaz'


class Sentence:

    def __init__(self):
        self.words = []
        self.newlyLabeled = []  # list of pairs of the form (sentence, score)
        self.fee = None

        # JSON data
        self.depParse = None
        self.target = None
        self.text = None
        self.luName = None
        self.luID = None
        self.frameId = None
        self.annotations = None

    def add_word(self, w):
        self.words.append(w)

    def add_sentence(self, pair):
        self.newlyLabeled.append(pair)

    def print_sentence(self):
        for i in range(len(self.words)):
            self.words[i].print_word()
        print()

    def to_string(self):
        s = "["
        for word in self.words:
            s += word.to_string() + '\n'
        s += "]"
        return s