
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
    try:
        outfile = f[:-4] + '.txt'
        if outfile in os.listdir(outdir):
            print('file already generated, skipping')
            continue
        else:
            doc = folia.Document(file = foliadir + f, encoding = 'utf-8')
            features = ft.extract_words(doc)
            with open(outfile, 'w', encoding = 'utf-8') as f_out:
                f_out.write(' '.join(features))
    except:
        #exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Error parsing doc', foliadir + f)
