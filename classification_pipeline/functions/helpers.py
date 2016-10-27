

import ucto


def tokenize_lines(lines, config, strip_punctuation = False, language = 'nl'):

    tokenizer = ucto.Tokenizer(config)
    
    tokenized = []
    for line in lines:
        tokens = []
        tokenizer.process(line)
        for token in tokenizer:
            if strip_punctuation:
                if token.type()=='PUNCT':
                    continue
            tokens.append(token.text)
        tokenized(tokens)

    return tokenized
