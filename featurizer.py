



class Featurizer:

    def __init__(self, foliadoc):
        self.doc = foliadoc
        self.features = []
        
    def extract_words(self, skip_punctuation = True, setname = 'current'):
        for p in self.doc.paragraphs():
            for s in p.sentences():
                for word in s.words():
                    if skip_punctuation and word.pos() == 'LET()':
                        continue
                    else:
                        self.features.append(word.text(setname))
    
    def extract_postags(self):
        pass


