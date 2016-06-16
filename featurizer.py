

class Featurizer:
    """
    Featurizer
    =====
    class to extract features from a folia-document
    
    Parameters
    -----
    skip_punctuation : Boolean
        choose to include punctuation or not
    lowercase : Boolean
        choose to set characters to lowercase
    setname : str
        the name of the word set           

    """

    def __init__(self, skip_punctuation = True, lowercase = True, setname = 'current'):
        self.skip_punctuation = skip_punctuation
        self.lowercase = lowercase
        self.setname = setname
        
    def extract_words(self, foliadoc):
        """
        Word extractor
        =====
        Function to extract words
        
        Parameters
        -----
        foliadoc : folia.Document
            the document in folia.Document format   

        Returns
        -----
        features : list
            the extracted words of type str
        
        """
        features = []
        for word in foliadoc.words():
            if self.skip_punctuation and hasattr(word, 'pos'):
                if word.pos() == 'LET()':
                    continue
            else:
                word_str = word.text(self.setname)
                if self.lowercase:
                    word_str = word_str.lower()
            features.append((word_str, pos))
        return features
        
#    def extract_words(self, foliadoc):
        """
        Word extractor
        =====
        Function to extract all words from a folia document, in the original order
        
        Parameters
        -----
        foliadoc : folia.Document
        
        Returns
        -----
        word_features : list
            list of all words in the original order

        """
#        features = self.extract_features(foliadoc)
#        word_features = [feature[0] for feature in features] 
#        return word_features
    
#    def extract_postags(self, foliadoc):
        """
        POS extractor
        =====
        Function to extract all pos tags from a folia document, in the original order
        
        Parameters
        -----
        foliadoc : folia.Document
        
        Returns
        -----
        pos_features : list
            list of all pos tags in the original order
        
        """
#        features = self.extract_features(foliadoc)
#        pos_features = [feature[1] for feature in features]
#        return pos_features
