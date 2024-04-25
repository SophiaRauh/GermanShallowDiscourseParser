# -*- coding: utf-8 -*-

import dill as pickle
import codecs
import configparser
import os
import re
import sys
import time
from collections import defaultdict

import numpy
import stanza
from nltk import word_tokenize
from nltk.tree.tree import Tree
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# custom modules
import DimLexParser
import PCCParser
import utils


class ExplicitSenseClassifier():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.nlp = stanza.Pipeline(lang='de',
                                   processors='tokenize,pos,mwt,constituency',
                                   package='default_accurate',
                                   tokenize_pretokenized=True,
                                   download_method=None)
        self.labelencodict = {}
        self.labeldecodict = {}
        self.maxencid = 42
        if os.path.exists(os.path.join(os.getcwd(),
                                       'bert_client_encodings.pickle')):
            self.bertmap = pickle.load(codecs.open(
                os.path.join(os.getcwd(), 'bert_client_encodings.pickle'),
                'rb'))
        else:
            self.bertmap = {}
        if os.path.exists(os.path.join(os.getcwd(), 'pcc_memorymap.pickle')):
            self.parsermap = pickle.load(codecs.open(
                os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'rb'))
        else:
            self.parsermap = {}

    def encode(self, val):
        if val in self.labelencodict:
            return self.labelencodict[val]
        else:
            self.labelencodict[val] = self.maxencid
            self.labeldecodict[self.maxencid] = val
            self.maxencid += 1
            return self.labelencodict[val]

    def decode(self, val):
        return self.labeldecodict[val]

    def getFeatures(self, rel):
        # syntactic features
        sentence = rel.connectiveTokens[0].fullSentence
        tokens = sentence.split()
        ptree = None
        if sentence in self.parsermap:
            ptree = self.parsermap[sentence]
        else:
            sent = sentence.replace('(', '[')
            sent = sent.replace(')', ']')
            doc = self.nlp(sent)
            ptree = Tree.fromstring(str(doc.sentences[0].constituency))
            self.parsermap[sentence] = ptree

        feat = ['_']*12
        match_positions = None
        # continuous connective
        if utils.iscontinuous([int(x.tokenId) for x in rel.connectiveTokens]):
            if utils.contains_sublist(tokens,
                                      [x.token for x in rel.connectiveTokens]):
                match_positions = utils.get_match_positions(
                    tokens, [x.token for x in rel.connectiveTokens])
                if len(rel.connectiveTokens) == 1:
                    for position in match_positions:
                        feat = utils.getFeaturesFromTreeCont(
                            ptree, position, rel.connectiveTokens[0].token)
                elif len(rel.connectiveTokens) > 1:
                    for startposition in match_positions:
                        positions = list(
                            range(startposition, startposition+len(
                                rel.connectiveTokens)))
                        feat = utils.getFeaturesFromTreeCont(
                            ptree, list(positions), tuple(
                                [x.token for x in rel.connectiveTokens]))
        else:  # discontinuous connective
            if utils.contains_discont_sublist(
                    tokens, [x.token for x in rel.connectiveTokens]):
                match_positions = utils.get_discont_match_positions(
                    tokens, [x.token for x in rel.connectiveTokens])
                feat = utils.getFeaturesFromTreeDiscont(
                    ptree, match_positions,
                    tuple([x.token for x in rel.connectiveTokens]))

        synfeats = [self.encode(v) for v in feat]

        # bert features
        conn = ' '.join([x.token for x in rel.connectiveTokens])
        intarg = ' '.join([x.token for x in rel.intArgTokens])
        extarg = ' '.join([x.token for x in rel.extArgTokens])
        leftarg = extarg  # unmarked order
        rightarg = intarg
        # marked order
        if rel.intArgTokens[-1].tokenId < rel.extArgTokens[0].tokenId:
            leftarg = intarg
            rightarg = extarg
        sentence = rel.connectiveTokens[0].fullSentence
        if tuple(tuple([leftarg, rightarg, conn])) in self.bertmap:
            enc = self.bertmap[tuple(tuple([leftarg, rightarg, conn]))]
        else:
            enc = self.model.encode([leftarg, rightarg, conn])
        self.bertmap[tuple(tuple([leftarg, rightarg, conn]))] = enc

        return synfeats, numpy.concatenate(enc)

    # second arg is to use only train filtes in cross-evaluation setup
    # (empty by default)
    def train(self, trainfiles=[]):

        start = time.time()
        sys.stderr.write(
            'INFO: Starting training of explicit sense classifier...\n')

        connectivefiles = [x for x in utils.listfolder(
            os.path.join(self.config['PCC']['pccdir'], 'connectives'))
            # filtering out temp/hidden files that may be there
            if re.search(r'maz-\d+.xml', x)]
        syntaxfiles = [x for x in utils.listfolder(
            os.path.join(self.config['PCC']['pccdir'], 'syntax'))
            if re.search(r'maz-\d+.xml', x)]

        fd = defaultdict(lambda: defaultdict(str))
        fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
        fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')
        # getting this over the training data, to override with the
        # most frequent sense in post-processing if predicted sense
        # does not match with dimlex
        self.conn2mostfrequent = defaultdict(lambda: defaultdict(int))

        X_train_bert = []
        X_train_syn = []
        y_train = []

        # filtering out test files if a list of train fileids is specified
        if trainfiles:
            fd = {f: fd[f] for f in fd if f in trainfiles}
        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(
                fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'explicit':
                    bertfeats, synfeats = self.getFeatures(rel)
                    X_train_bert.append(bertfeats)
                    X_train_syn.append(synfeats)
                    y_train.append(rel.sense)
                    self.conn2mostfrequent[tuple(
                        [x.token for x
                         in rel.connectiveTokens])][rel.sense] += 1

        # overwriting memory maps (commented out because the ones
        # uploaded to github contain all training input)
        # pickle.dump(self.bertmap, codecs.open(
        #     os.path.join(os.getcwd(), 'bert_client_encodings.pickle'), 'wb'))
        # pickle.dump(self.parsermap, codecs.open(
        #     os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'wb'))

        rf = RandomForestClassifier(class_weight='balanced', n_estimators=1000,
                                    random_state=42)
        mlp = MLPClassifier(max_iter=500, random_state=42)

        clfs = [rf, mlp]
        X_train = [X_train_syn, X_train_bert]
        self.le = LabelEncoder()
        self.le.fit(y_train)
        self.clfs = [clf.fit(X, y_train) for clf, X in zip(clfs, X_train)]

        # get conn2senses from dimlex for manual overriding as
        # post-processing step
        self.conn2senses = {}
        dimlex = DimLexParser.parseXML(
            os.path.join(self.config['DiMLex']['dimlexdir'], 'DimLex.xml'))
        for entry in dimlex:
            altdict = entry.alternativeSpellings
            senses = entry.sense2Probs.keys()
            # canonical form is always in list of alt spellings
            for item in altdict:
                tupl = tuple(word_tokenize(item))
                self.conn2senses[tupl] = senses

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        sys.stderr.write(
            'INFO: Done training explicit sense classifier'
            '...({:0>2}:{:0>2}:{:0>2})\n'.format(int(hours),
                                                 int(minutes), int(seconds)))

        # pickle.dump(self.clfs, codecs.open(
        # 'explicit_sense_classifier.pickle', 'wb'))
        # pickle.dump(self.le, codecs.open('explicit_label_encoder.pickle',
        # 'wb'))
        # sys.stderr.write(
        # 'INFO: Saved labelencoder to explicit_label_encoder.pickle.\n')
        # sys.stderr.write(
        # 'INFO: Saved classifier to explicit_sense_classifier.pickle.\n')

    def load(self):

        if not os.path.exists(
                os.path.join(os.getcwd(), 'explicit_sense_classifier.pickle')):
            return 'ERROR: explicit_sense_classifier.pickle not found.\n'
        if not os.path.exists(
                os.path.join(os.getcwd(), 'explicit_label_encoder.pickle')):
            return 'ERROR: explicit_label_encoder.pickle not found.\n'

        self.clfs = pickle.load(codecs.open('explicit_sense_classifier.pickle',
                                            'rb'))
        self.le = pickle.load(codecs.open('explicit_label_encoder.pickle',
                                          'rb'))

        self.conn2senses = {}
        dimlex = DimLexParser.parseXML(
            os.path.join(self.config['DiMLex']['dimlexdir'], 'DimLex.xml'))
        for entry in dimlex:
            altdict = entry.alternativeSpellings
            senses = entry.sense2Probs.keys()
            # canonical form is always in list of alt spellings
            for item in altdict:
                tupl = tuple(word_tokenize(item))
                self.conn2senses[tupl] = senses

    def getGoldSenses(self, testfiles):

        connectivefiles = [x for x in utils.listfolder(
            os.path.join(self.config['PCC']['pccdir'], 'connectives'))
            # filtering out temp/hidden files that may be there
            if re.search(r'maz-\d+.xml', x)]
        syntaxfiles = [x for x in utils.listfolder(
            os.path.join(self.config['PCC']['pccdir'], 'syntax'))
            if re.search(r'maz-\d+.xml', x)]

        fd = defaultdict(lambda: defaultdict(str))
        fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
        fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')

        # taking test files only
        fd = {f: fd[f] for f in fd if f in testfiles}
        goldsenses = []
        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(
                fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'explicit':
                    goldsenses.append(rel)

        return goldsenses

    def evaluate_gold(self, testfiles, f2gold):

        X_test_syn = []
        X_test_bert = []
        y_test = []
        candidates = []
        total = 0
        correct = 0

        for f in tqdm(f2gold):
            if f in testfiles:
                relations, sents, tokens = f2gold[f]
                for rel in tqdm(relations):
                    if rel.relationType == 'explicit':
                        sentence = rel.connective[0].fullSentence
                        tokens = sentence.split()
                        ptree = None
                        if sentence in self.parsermap:
                            ptree = self.parsermap[sentence]
                        else:
                            sent = sentence.replace('(', '[')
                            sent = sent.replace(')', ']')
                            doc = self.nlp(sent)
                            ptree = Tree.fromstring(str(
                                doc.sentences[0].constituency))
                            self.parsermap[sentence] = ptree

                        feat = ['_']*12
                        match_positions = None
                        if utils.iscontinuous(
                                # continuous connective
                                [int(x.tokenId) for x in rel.connective]):
                            if utils.contains_sublist(
                                    tokens, [x.token for x in rel.connective]):
                                match_positions = utils.get_match_positions(
                                    tokens, [x.token for x in rel.connective])
                                if len(rel.connective) == 1:
                                    for position in match_positions:
                                        feat = utils.getFeaturesFromTreeCont(
                                            ptree, position,
                                            rel.connective[0].token)
                                elif len(rel.connective) > 1:
                                    for startposition in match_positions:
                                        positions = list(
                                            range(startposition,
                                                  startposition+len(
                                                      rel.connective)))
                                        feat = utils.getFeaturesFromTreeCont(
                                            ptree, list(positions), tuple(
                                                [x.token for x
                                                 in rel.connective]))
                        else:  # discontinuous connective
                            if utils.contains_discont_sublist(
                                    tokens, [x.token for x in rel.connective]):
                                match_positions =\
                                    utils.get_discont_match_positions(
                                        tokens,
                                        [x.token for x in rel.connective])
                                feat = utils.getFeaturesFromTreeDiscont(
                                    ptree, match_positions,
                                    tuple([x.token for x in rel.connective]))
                        synfeats = [self.encode(v) for v in feat]

                        conn = ' '.join([x.token for x in rel.connective])
                        intarg = ' '.join([x.token for x in rel.arg2])
                        extarg = ' '.join([x.token for x in rel.arg1])
                        leftarg = extarg  # unmarked order
                        rightarg = intarg
                        # marked order
                        if rel.arg2[-1].tokenId < rel.arg1[0].tokenId:
                            leftarg = intarg
                            rightarg = extarg
                        enc = None
                        if tuple(tuple(
                                [leftarg, rightarg, conn])) in self.bertmap:
                            enc = self.bertmap[tuple(tuple([leftarg,
                                                            rightarg, conn]))]
                        else:
                            enc = self.model.encode(
                                [' '.join(utils.bertclient_safe(
                                    leftarg.split())),
                                 ' '.join(utils.bertclient_safe(
                                     rightarg.split())),
                                 ' '.join(utils.bertclient_safe(
                                     conn.split()))])
                        bertfeats = numpy.concatenate(enc)

                        X_test_syn.append(synfeats)
                        X_test_bert.append(bertfeats)
                        y_test.append(rel.sense)
                        candidates.append(tuple(
                            [x.token for x in rel.connective]))

        if candidates:
            X_test = [X_test_bert, X_test_syn]
            pred1 = numpy.asarray([clf.predict_proba(X) for clf, X in zip(
                self.clfs, X_test)])
            pred2 = numpy.average(pred1, axis=0)
            pred = numpy.argmax(pred2, axis=1)

            assert len(pred) == len(candidates)

            pred = self.le.inverse_transform(pred)

            # checking predicted sense with dimlex and overriding if
            # not matching:
            # one way to speed up this code is to take unambiguous
            # sense conns from dimlex right away, without the prediction part
            for i, t in enumerate(zip(pred, candidates)):
                p, s = t
                if s in self.conn2senses:
                    if p not in self.conn2senses[s]:
                        if len(self.conn2senses[s]) == 1:
                            pred[i] = list(self.conn2senses[s])[0]
                        else:
                            if s in self.conn2mostfrequent:
                                top = sorted(
                                    self.conn2mostfrequent[s].items(),
                                    key=lambda x: x[1], reverse=True)[0][0]
                                pred[i] = top

            detailed_f1 = f1_score(pred, y_test, average='weighted')
            second_level_f1 = f1_score(
                ['.'.join(x.split('.')[:2]) for x in pred],
                ['.'.join(x.split('.')[:2]) for x in y_test],
                average='weighted')
            first_level_f1 = f1_score([x.split('.')[0] for x in pred],
                                      [x.split('.')[0] for x in y_test],
                                      average='weighted')
            for pair in zip(pred, y_test):
                total += 1
                if pair[0] == pair[1]:
                    correct += 1

        return detailed_f1, second_level_f1, first_level_f1, total, correct

    def evaluate_pred(self, pred_relations, gold_relations):
        tot = 0
        dcor = 0
        scor = 0
        fcor = 0
        for grel in gold_relations:
            tot += 1
            grel_conn = sorted([int(x.tokenId) for x
                                in grel.connectiveTokens])  # -1 bei tokenId
            for prel in pred_relations:
                prel_conn = sorted([x.tokenId for x in prel.connective])

                if prel_conn == grel_conn:
                    if grel.sense == prel.sense:
                        dcor += 1
                    if grel.sense.split('.')[:2] == prel.sense.split('.')[:2]:
                        scor += 1
                    if grel.sense.split('.')[0] == prel.sense.split('.')[0]:
                        fcor += 1

        return tot, dcor, scor, fcor

    def predict(self, relations):
        X_test_syn = []
        X_test_bert = []
        candidates = []
        for rel in relations:
            sentence = rel.connective[0].fullSentence
            tokens = sentence.split()
            ptree = None
            sent = sentence.replace('(', '[')
            sent = sent.replace(')', ']')
            doc = self.nlp(sent)
            ptree = Tree.fromstring(str(
                doc.sentences[0].constituency))
            self.parsermap[sentence] = ptree

            feat = ['_']*12
            match_positions = None
            # continuous connective
            if utils.iscontinuous([int(x.tokenId) for x in rel.connective]):
                if utils.contains_sublist(tokens,
                                          [x.token for x in rel.connective]):
                    match_positions = utils.get_match_positions(
                        tokens, [x.token for x in rel.connective])
                    if len(rel.connective) == 1:
                        for position in match_positions:
                            feat = utils.getFeaturesFromTreeCont(
                                ptree, position, rel.connective[0].token)
                    elif len(rel.connective) > 1:
                        for startposition in match_positions:
                            positions = list(
                                range(startposition,
                                      startposition+len(rel.connective)))
                            feat = utils.getFeaturesFromTreeCont(
                                ptree, list(positions),
                                tuple([x.token for x in rel.connective]))
            else:  # discontinuous connective
                if utils.contains_discont_sublist(
                        tokens, [x.token for x in rel.connective]):
                    match_positions = utils.get_discont_match_positions(
                        tokens, [x.token for x in rel.connective])
                    feat = utils.getFeaturesFromTreeDiscont(
                        ptree, match_positions,
                        tuple([x.token for x in rel.connective]))
            synfeats = [self.encode(v) for v in feat]

            conn = ' '.join([x.token for x in rel.connective])
            intarg = ' '.join([x.token for x in rel.arg2])
            extarg = ' '.join([x.token for x in rel.arg1])
            leftarg = extarg  # unmarked order
            rightarg = intarg
            try:
                if rel.arg2[-1].tokenId < rel.arg1[0].tokenId:  # marked order
                    leftarg = intarg
                    rightarg = extarg
            except IndexError:
                pass  # one of the two (or both) not found/empty
            enc = self.model.encode(
                [' '.join(utils.bertclient_safe(leftarg)),
                 ' '.join(utils.bertclient_safe(rightarg)),
                 ' '.join(utils.bertclient_safe(conn))])
            bertfeats = numpy.concatenate(enc)

            X_test_syn.append(synfeats)
            X_test_bert.append(bertfeats)
            candidates.append(tuple([x.token for x in rel.connective]))

        if candidates:
            X_test = [X_test_bert, X_test_syn]
            pred1 = numpy.asarray(
                [clf.predict_proba(X) for clf, X in zip(self.clfs, X_test)])
            pred2 = numpy.average(pred1, axis=0)
            pred = numpy.argmax(pred2, axis=1)

            assert len(pred) == len(candidates)

            pred = self.le.inverse_transform(pred)

            # checking predicted sense with dimlex and overriding if not
            # matching:
            # one way to speed up this code is to take unambiguous sense conns
            # from dimlex right away, without the prediction part
            for i, t in enumerate(zip(pred, candidates)):
                p, s = t
                if s in self.conn2senses:
                    if p not in self.conn2senses[s]:
                        if len(self.conn2senses[s]) == 1:
                            pred[i] = list(self.conn2senses[s])[0]
                        else:
                            if s in self.conn2mostfrequent:
                                top = sorted(self.conn2mostfrequent[s].items(),
                                             key=lambda x: x[1],
                                             reverse=True)[0][0]
                                pred[i] = top

            for pair in zip(relations, pred):
                rel, prediction = pair
                rel.addSense(prediction)


if __name__ == '__main__':
    exp = ExplicitSenseClassifier()
    exp.train()
