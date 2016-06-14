
import sys
import os

from pynlpl.formats import folia
import featurizer

foliadir = sys.argv[1]
outdir = sys.argv[2]

files = os.listdir(foliadir)

#ft = featurizer.Featurizer(lowercase = False, skip_punctuation = False, setname = 'piccl')
ft = featurizer.Featurizer()

for f in files:
    print(f, foliadir + f)
    doc = folia.Document(file = foliadir + f, encoding = 'utf-8')
    features = ft.extract_words(doc)
    outfile = outdir + f[:-4] + '.txt'
    with open(outfile, 'w', encoding = 'utf-8') as f_out:
        f_out.write(' '.join(features))
