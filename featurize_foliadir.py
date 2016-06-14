
import sys
import os

from pynlpl.formats import folia
import featurizer

foliadir = sys.argv[1]
outdir = sys.argv[2]

files = os.listdir(foliadir)

for f in files:
    print(f, foliadir + f)
    doc = folia.Document(file = foliadir + f, encoding = 'utf-8')
    ft = featurizer.Featurizer(doc)
    ft.extract_words()
    outfile = outdir + f[:-4] + '.txt'
    with open(outfile, 'w', encoding = 'utf-8') as f_out:
        f_out.write(' '.join(ft.features))


    
    

