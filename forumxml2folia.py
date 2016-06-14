import sys
import os
from datetime import datetime
import lxml.etree
from pynlpl.formats import folia



def forumxml2folia(inputfilename, outputfilename):
    #read the input document
    forumdoc = lxml.etree.parse(inputfilename).getroot()

    #determine a good ID to use for everything
    forumtype = forumdoc.xpath('/forum')[0].attrib['type']
    thread_id = forumdoc.xpath('/forum/thread')[0].attrib['id']
    doc_id = forumtype + '.' + thread_id


    #start an empty FoLiA document
    doc = folia.Document(id=doc_id)

    #set document metadata
    doc.metadata['title'] = forumdoc.xpath('/forum/thread/title')[0].text
    doc.metadata['category'] = forumdoc.xpath('/forum/thread/category')[0].text

    #declare event (set definition doesn't exist yet but doesn't matter)
    doc.declare(folia.Event, "https://raw.githubusercontent.com/LanguageMachines/quoll/master/setdefinitions/forum.xml")

    #add text container and an event for the thread
    textbody = folia.Text(doc, id=doc_id+'.text')
    thread = textbody.add(folia.Event, cls="thread", id=doc_id+'.thread')

    #process all posts
    for postnode in forumdoc.xpath('//post'):
        timestamp = postnode.xpath('timestamp')[0].text
        author = postnode.xpath('author')[0].text
        post = thread.add(folia.Event, cls="post", id=doc_id+'.post.' + postnode.attrib['id'],begindatetime=datetime.strptime(timestamp, '%d-%m-%Y %H:%M'), actor=author)
        post.add(folia.TextContent, postnode.xpath('body')[0].text)

    #add the text body to the document
    doc.append(textbody)
    
    #save everything
    doc.save(outputfilename)


if __name__ == '__main__':
    try:
        inputfilename = sys.argv[1]
    except:
        print("Usage: forumxml2folia.py [inputfile]",file=sys.stderr)
    outputfilename = inputfilename.replace('.xml','') + '.folia.xml'
    forumxml2folia(inputfilename, outputfilename)

