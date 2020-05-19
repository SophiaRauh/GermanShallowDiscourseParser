#!/usr/bin/python3

import os
import sys
import json
import codecs
#import configparser
#from nltk.parse import stanford
import dill as pickle
from spacy.lang.de import German

# custom modules
import utils
import ConnectiveClassifier
import ExplicitArgumentExtractor
import ExplicitSenseClassifier
import ImplicitSenseClassifier

nlp = German()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

"""
class Parser:

    def __init__(self):
        # doing this in sub-modules now. Check back in later to see if there's a central solution
        #self.config = configparser.ConfigParser()
        #self.config.read('config.ini')
        #os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        #os.environ['STANFORD_PARSER'] = self.config['lexparser']['parserdir']
        #os.environ['STANFORD_MODELS'] = self.config['lexparser']['parserdir']
        #os.environ['CLASSPATH'] = '%s/stanford-parser.jar' % self.config['lexparser']['parserdir']
        #self.lexparser = stanford.StanfordParser(model_path='edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz')
"""

class Token:

    def __init__(self, token, sentenceId, sentenceTokenId):
        self.token = token.text
        self.tokenId = token.i
        self.sentenceId = sentenceId
        self.sentenceTokenId = sentenceTokenId
        self.span = tuple((token.idx, token.idx+len(token.text)))

    def setConnective(self):
        self.isConnective = True

    def setMultiToken(self, y):
        if not hasattr(self, 'multiTokenIds'):
            self.multiTokenIds = []
        self.multiTokenIds.append(y)
    def setFullSentence(self, val):
        self.fullSentence = val

class Relation:

    def __init__(self, _id, _type, docId):
        self.relationId = _id
        self.relationType = _type
        self.docId = docId
        self.connective = []
        self.arg1 = []
        self.arg2 = []
        
    def addConnectiveToken(self, token):
        self.connective.append(token)
    def addIntArgToken(self, token):
        self.arg2.append(token)
    def addExtArgToken(self, token):
        self.arg1.append(token)
    def addSense(self, sense):
        self.sense = sense
        
def custom_tokenize(inp):

    # using spacy sentencizer/tokenizer since most(all?) nltk ones replace double quotes (and some other chars: https://www.nltk.org/_modules/nltk/tokenize/treebank.html)
    doc = nlp(inp)
    sents = {}
    tokens = {}
    for si, sent in enumerate(doc.sents):
        senttokens = []
        fullsent = ' '.join([t.text for t in sent])
        for ti, token in enumerate(sent):
            t = Token(token, si, ti)
            t.setFullSentence(fullsent)
            senttokens.append(t)
            tokens[t.tokenId] = t
        sents[si] = senttokens
        
    return sents, tokens
    
    

if __name__ == '__main__':


    cc = ConnectiveClassifier.ConnectiveClassifier()
    eae = ExplicitArgumentExtractor.ExplicitArgumentExtractor()
    esc = ExplicitSenseClassifier.ExplicitSenseClassifier()
    isc = ImplicitSenseClassifier.ImplicitSenseClassifier()

    #cc.train()
    #eae.train()
    #esc.train()
    #isc.train()
    
    """
    inp = 'Wie schwierig es ist, in dieser Region einen Ausbildungsplatz zu finden, haben wir an dieser und anderer Stelle oft und ausführlich bewertet. Trotzdem bemühen sich Unternehmen sowie die Industrie- und Handelskammer Potsdam den Schulabgängern Wege in die Ausbildung aufzuzeigen. Und Beispielsweise gibt es ein mit entweder dies oder das, und dazu gibt es noch anstatt dass aapjes. Entweder bezahlen für die Schülung, oder später im Arsch gehen. Und das ist ein guter erster Schritt. Das weiß jedes Kind, aber nicht jeder hält sich daran. Das Schlimmste aber ist, dass noch heute versucht wird, zu mauscheln. Hier gibt es ein Satz. Hier gibt es noch ein Satz.' 

    sents, tokens = custom_tokenize(inp)
    cc.predict(sents)

    # populating list of relations, starting point are explicits/connectives
    relations = []
    _id = 1
    already_processed = [] # for phrasal connectives...
    for sid in sents:
        for i, token in enumerate(sents[sid]):
            if hasattr(token, 'isConnective') and not token.tokenId in already_processed:
                rel = Relation(_id, 'Explicit', 'dummy')
                rel.addConnectiveToken(token)
                if hasattr(token, 'multiTokenIds'):
                    for ot in token.multiTokenIds:
                        rel.addConnectiveToken(tokens[ot])
                        already_processed.append(ot)
                relations.append(rel)
                _id += 1

    # for dev/debugging, pickling result of the above
    pickle.dump(sents, codecs.open('sents_debug.pickle', 'wb'))
    pickle.dump(tokens, codecs.open('tokens_debug.pickle', 'wb'))
    pickle.dump(relations, codecs.open('relations_debug.pickle', 'wb'))
    """
    #eae.predict(relations, sents, tokens)
    #esc.predict(relations)

    #pickle.dump(sents, codecs.open('sents_debug.pickle', 'wb'))
    #pickle.dump(tokens, codecs.open('tokens_debug.pickle', 'wb'))
    #pickle.dump(relations, codecs.open('relations_debug.pickle', 'wb'))

    sents = pickle.load(codecs.open('sents_debug.pickle', 'rb'))
    tokens = pickle.load(codecs.open('tokens_debug.pickle', 'rb'))
    relations = pickle.load(codecs.open('relations_debug.pickle', 'rb'))

    """
    newrels = isc.predict(relations, sents)
    maxrelid = max([x.relationId for x in relations])
    for nr in newrels:
        r = Relation(maxrelid+1, 'Implicit', 'dummy')
        maxrelid += 1
        for t in nr[0]:
            r.addExtArgToken(t)
        for t in nr[1]:
            r.addIntArgToken(t)
        r.addSense(nr[2])
        relations.append(r)
    """
    
    # TODO:
    # then wrap in flask/dockerise, such that Olha can use it,
    # then evaluation...
                
    #"""
    for rel in relations:
        print('relid:', rel.relationId)
        print('type:', rel.relationType)
        print('conns:', [x.token for x in rel.connective])
        print('arg1:', [x.token for x in rel.arg1])
        print('arg2:', [x.token for x in rel.arg2])
        print('sense:', rel.sense)
        print()
    #"""
    
    
