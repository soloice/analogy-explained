import codecs


class QuestionWords(object):
    def __init__(self, file_name="../data/questions-words.txt", to_lower=True, verbose=False):
        self.question_words_types = {}
        self.question_words_problems = {}
        self.question_words_pairs = {}
        key, prototype = None, {}
        with codecs.open(file_name, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(":"):
                    key = line[1:].strip()
                    self.question_words_problems[key] = []
                    self.question_words_pairs[key] = []
                    if verbose:
                        print(key)
                else:
                    a, b, c, d = line.split()
                    if to_lower:
                        a, b, c, d = a.lower(), b.lower(), c.lower(), d.lower()
                    self.question_words_problems[key].append((a, b, c, d))
                    if (a, b) not in self.question_words_pairs[key]:
                        self.question_words_pairs[key].append((a, b))
                    if (c, d) not in self.question_words_pairs[key]:
                        self.question_words_pairs[key].append((c, d))
        for k in self.question_words_problems.keys():
            self.question_words_types[k] = len(self.question_words_problems[k]), len(self.question_words_pairs[k])

    def show(self):
        for k in self.question_words_problems.keys():
            print(k, self.question_words_types[k])
            v = self.question_words_problems[k]
            print("Sample questions:")
            print(v[0])
            print(v[1])
            v2 = self.question_words_pairs[k]
            print("Sample pairs:")
            print(v2[0])
            print(v2[1])

if __name__ == "__main__":
    qw = QuestionWords("../data/questions-words.txt")
    qw.show()

# question types:
# capital-common-countries
# capital-world
# currency
# city-in-state
# family
# gram1-adjective-to-adverb
# gram2-opposite
# gram3-comparative
# gram4-superlative
# gram5-present-participle
# gram6-nationality-adjective
# gram7-past-tense
# gram8-plural
# gram9-plural-verbs
#
# gram7-past-tense (1560, 40)
# Sample questions:
# ('dancing', 'danced', 'decreasing', 'decreased')
# ('dancing', 'danced', 'describing', 'described')
# Sample pairs:
# ('dancing', 'danced')
# ('decreasing', 'decreased')
# gram5-present-participle (1056, 33)
# Sample questions:
# ('code', 'coding', 'dance', 'dancing')
# ('code', 'coding', 'debug', 'debugging')
# Sample pairs:
# ('code', 'coding')
# ('dance', 'dancing')
# currency (866, 30)
# Sample questions:
# ('Algeria', 'dinar', 'Angola', 'kwanza')
# ('Algeria', 'dinar', 'Argentina', 'peso')
# Sample pairs:
# ('Algeria', 'dinar')
# ('Angola', 'kwanza')
# gram2-opposite (812, 29)
# Sample questions:
# ('acceptable', 'unacceptable', 'aware', 'unaware')
# ('acceptable', 'unacceptable', 'certain', 'uncertain')
# Sample pairs:
# ('acceptable', 'unacceptable')
# ('aware', 'unaware')
# city-in-state (2467, 68)
# Sample questions:
# ('Chicago', 'Illinois', 'Houston', 'Texas')
# ('Chicago', 'Illinois', 'Philadelphia', 'Pennsylvania')
# Sample pairs:
# ('Chicago', 'Illinois')
# ('Houston', 'Texas')
# gram9-plural-verbs (870, 30)
# Sample questions:
# ('decrease', 'decreases', 'describe', 'describes')
# ('decrease', 'decreases', 'eat', 'eats')
# Sample pairs:
# ('decrease', 'decreases')
# ('describe', 'describes')
# family (506, 23)
# Sample questions:
# ('boy', 'girl', 'brother', 'sister')
# ('boy', 'girl', 'brothers', 'sisters')
# Sample pairs:
# ('boy', 'girl')
# ('brother', 'sister')
# gram4-superlative (1122, 34)
# Sample questions:
# ('bad', 'worst', 'big', 'biggest')
# ('bad', 'worst', 'bright', 'brightest')
# Sample pairs:
# ('bad', 'worst')
# ('big', 'biggest')
# gram1-adjective-to-adverb (992, 32)
# Sample questions:
# ('amazing', 'amazingly', 'apparent', 'apparently')
# ('amazing', 'amazingly', 'calm', 'calmly')
# Sample pairs:
# ('amazing', 'amazingly')
# ('apparent', 'apparently')
# gram6-nationality-adjective (1599, 41)
# Sample questions:
# ('Albania', 'Albanian', 'Argentina', 'Argentinean')
# ('Albania', 'Albanian', 'Australia', 'Australian')
# Sample pairs:
# ('Albania', 'Albanian')
# ('Argentina', 'Argentinean')
# gram8-plural (1332, 37)
# Sample questions:
# ('banana', 'bananas', 'bird', 'birds')
# ('banana', 'bananas', 'bottle', 'bottles')
# Sample pairs:
# ('banana', 'bananas')
# ('bird', 'birds')
# capital-world (4524, 116)
# Sample questions:
# ('Abuja', 'Nigeria', 'Accra', 'Ghana')
# ('Abuja', 'Nigeria', 'Algiers', 'Algeria')
# Sample pairs:
# ('Abuja', 'Nigeria')
# ('Accra', 'Ghana')
# capital-common-countries (506, 23)
# Sample questions:
# ('Athens', 'Greece', 'Baghdad', 'Iraq')
# ('Athens', 'Greece', 'Bangkok', 'Thailand')
# Sample pairs:
# ('Athens', 'Greece')
# ('Baghdad', 'Iraq')
# gram3-comparative (1332, 37)
# Sample questions:
# ('bad', 'worse', 'big', 'bigger')
# ('bad', 'worse', 'bright', 'brighter')
# Sample pairs:
# ('bad', 'worse')
# ('big', 'bigger')
