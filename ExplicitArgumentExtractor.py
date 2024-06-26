# -*- coding: utf-8 -*-

import codecs
import configparser
import dill as pickle
import os
import re
import string
import sys
import time
from collections import defaultdict

import numpy
import stanza
from nltk.tree import ParentedTree
from nltk.tree.tree import Tree
from sklearn.ensemble import RandomForestClassifier

# custom modules
import PCCParser
import utils


class ExplicitArgumentExtractor:

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.nlp = stanza.Pipeline(lang='de',
                                   processors='tokenize,pos,mwt,constituency',
                                   package='default_accurate',
                                   tokenize_pretokenized=True,
                                   download_method=None)
        if os.path.exists(os.path.join(os.getcwd(), 'pcc_memorymap.pickle')):
            self.parsermap = pickle.load(codecs.open(
                os.path.join(os.getcwd(), 'pcc_memorymap.pickle'), 'rb'))
        else:
            self.parsermap = {}

        self.labelencodict = {}
        self.labeldecodict = {}
        self.maxencid = 42

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

    def getIntArg(self, tree, rel, tokens):

        refcon = rel.connective[0]
        leavenr = utils.getLeaveNr(refcon, tree)

        refconpos = tree.leaf_treeposition(leavenr)
        connnode = tree[refconpos[:-1]]
        refcontype = connnode.label()
        # print(refcontype)
        # TODO check these
        plain_tokens = []
        if refcontype == 'CCONJ':
            plain_tokens = utils.get_right_sibling(tree, leavenr, refcon)
        elif refcontype == 'ADV' or refcontype == 'SCONJ':
            plain_tokens = utils.get_parent_phrase_plus_phrase_after_comma(
                tree, leavenr, ['S', 'CS', 'VP'], refcon)
        else:
            plain_tokens = utils.get_parent_phrase(tree, leavenr,
                                                   ['S', 'CS', 'VP'], refcon)

        # because arg finding uses NLTK tree and positions may be off
        # due to brackets etc., getting back on track with actual Tokens here:
        id2token = {}
        intargtokens = []
        for i3, j3 in enumerate(refcon.fullSentence.split()):
            diff = refcon.sentenceTokenId - i3
            tokenId = refcon.tokenId - diff
            id2token[tokenId] = j3
        intargtokens = utils.matchPlainTokensWithIds(plain_tokens, id2token)
        if not intargtokens:  # stupid bracket replacing in nltk...
            replain_tokens = utils.bracketreplace(plain_tokens)
            intargtokens = utils.matchPlainTokensWithIds(replain_tokens,
                                                         id2token)

        # excluding the connective token(s) from intarg:
        intargtokens = [x for x in intargtokens
                        if x not in [y.tokenId for y in rel.connective]]
        # in PCC, intarg is what comes after second item for discont conns.
        # Guess this is a relatively arbitrary decision.
        if not utils.iscontinuous([x.tokenId for x in rel.connective]):
            intargtokens = [x for x in intargtokens
                            if x > rel.connective[-1].tokenId]
        # highly arbitrary as well, but in the PCC, final punctuation
        # signs are part of intarg
        if intargtokens:
            if intargtokens[-1] < len(tokens)-1:
                if tokens[intargtokens[-1]+1].token in string.punctuation:
                    intargtokens.append(tokens[intargtokens[-1]+1].tokenId)

        return intargtokens

    def getExtArg(self, tree, rel, tokens, sents):

        refcon = rel.connective[0]

        postag = tree.pos()[refcon.sentenceTokenId][1]
        ln = 'SOS' if refcon.sentenceTokenId == 0\
            else tree.pos()[refcon.sentenceTokenId-1][0]
        rn = 'EOS' if refcon.sentenceTokenId == len(tree.pos())\
            else tree.pos()[refcon.sentenceTokenId+1][0]
        lnpos = 'SOS' if refcon.sentenceTokenId == 0\
            else tree.pos()[refcon.sentenceTokenId-1][1]
        rnpos = 'EOS' if refcon.sentenceTokenId == len(tree.pos())\
            else tree.pos()[refcon.sentenceTokenId+1][1]
        nodePosition = tree.leaf_treeposition(refcon.sentenceTokenId)
        parent = tree[nodePosition[:-1]].parent()
        rootroute = utils.getPathToRoot(parent, [])
        feat = [' '.join([x.token for x in rel.connective]), postag, ln, rn,
                lnpos, rnpos, '-'.join(rootroute), refcon.sentenceTokenId]
        enc_feat = [self.encode(v) for v in feat]

        relative_position = self.sentposclf.predict(
            numpy.array(enc_feat).reshape(1, -1))

        # the way this works is that first all tokens from the
        # predicted sentenceId are taken. Then, if extarg is predicted
        # to be in the same sent, more restrictive filtering is applied
        # below. After intarg has been extracted, both are filtered
        # against each other again.
        temparg = []
        targetsid = refcon.sentenceId + relative_position[0]
        if targetsid in sents:
            for token in sents[targetsid]:
                temparg.append(token.tokenId)
        '''
        # the following is a more precise approach. Found, though, that
        just taking the entire predicted sentence (as per a few lines
        above) results in higher scores on the PCC. Later on, filtering
        with intarg tokens is done anyway, such that extarg is the
        mirror image of intarg (if both are in the same sentence).
        if relative_position == 0:
            samesentpos = self.samesentclf.predict(
                numpy.array(enc_feat).reshape(1, -1))
            if samesentpos[0] == 1: # extarg is predicted after (ref)conn
                # above, already the entire sent is taken by default,
                and filtered here, and against intarg later. This sct is
                a bit more restrictive and only takes the bit to the
                left/right (depending on samesentclf prediction) in the
                same S, CS or VP. May want to take out VP later if
                recall is too low. Or take right sibling (of the parent
                node) as well.
                labels = ['S', 'CS', 'VP']
                for i, node in enumerate(tree.pos()):
                    if i == leavenr:
                        nodePosition = tree.leaf_treeposition(i)
                        pn = utils.climb_tree(tree, nodePosition, labels)
                        # taking a guess here, which is that inside the
                        parent, the connective appears only once (taking
                        its first index below)... this bit is not
                        guaranteed to be correct
                        parent_tokens = [x[0] for x in pn.pos()]
                        ind = parent_tokens.index(refcon.token)
                        temparg = []
                        for i6, node in enumerate(pn.pos()):
                            if i6 > ind:
                                temparg.append(refcon.tokenId + (i6 - ind))

            elif samesentpos[0] == -1: # extarg is predicted before (ref)conn
                labels = ['S', 'CS', 'VP']
                for i, node in enumerate(tree.pos()):
                    if i == leavenr:
                        nodePosition = tree.leaf_treeposition(i)
                        pn = utils.climb_tree(tree, nodePosition, labels)
                        parent_tokens = [x[0] for x in pn.pos()]
                        ind = parent_tokens.index(refcon.token)
                        temparg = []
                        for i5, node in enumerate(pn.pos()):
                            if i5 < ind:
                                temparg.append(refcon.tokenId - (ind-i5))

            # adding final punctuation here if it's not part of arg (
            but the only trailing char in the sentence) (Mind
            indentation level here: only do this for same sentence cases)
            # assert refcon.fullSentence == ' '.join([x.token for x
                in sents[refcon.sentenceId]])
            if sents[refcon.sentenceId][-1].token in string.punctuation:
                temparg.append(sents[refcon.sentenceId][-1].tokenId)
        '''

        # in PCC, extarg is what comes before second item for discont conns.
        # Guess this is a relatively arbitrary decision. Hardcoding a fix here:
        if not utils.iscontinuous([x.tokenId for x in rel.connective]):
            temparg = [x for x in range(
                rel.connective[0].tokenId, rel.connective[-1].tokenId)]

        # filtering out connective tokens
        temparg = [x for x in temparg if x not in [y.tokenId for y
                                                   in rel.connective]]

        return temparg

    def train(self, trainfiles=[]):
        # second arg is to use only train filtes in cross-evaluation setup
        # (empty by default)
        start = time.time()
        # actually, the only thing being trained is/are the position
        # classifiers, since the rest is rule-based
        sys.stderr.write(
            'INFO: Starting training of explicit argument extractor...\n')

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

        X_train_pos = []
        y_train_pos = []
        X_train_samesent = []
        y_train_samesent = []

        # filtering out test files if a list of train fileids is specified
        if trainfiles:
            fd = {f: fd[f] for f in fd if f in trainfiles}
        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(
                fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'explicit':
                    connective = ' '.join(
                        [x.token for x in rel.connectiveTokens])
                    refcon = rel.connectiveTokens[0]
                    sentence = refcon.fullSentence
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
                    ptree = ParentedTree.convert(ptree)
                    postag = ptree.pos()[refcon.sentencePosition][1]
                    ln = 'SOS' if refcon.sentencePosition == 0\
                        else ptree.pos()[refcon.sentencePosition-1][0]
                    rn = 'EOS' if refcon.sentencePosition == len(ptree.pos())\
                        else ptree.pos()[refcon.sentencePosition+1][0]
                    lnpos = 'SOS' if refcon.sentencePosition == 0\
                        else ptree.pos()[refcon.sentencePosition-1][1]
                    rnpos = 'EOS' if refcon.sentencePosition == len(
                        ptree.pos())\
                        else ptree.pos()[refcon.sentencePosition+1][1]
                    nodePosition = ptree.leaf_treeposition(
                        refcon.sentencePosition)
                    parent = ptree[nodePosition[:-1]].parent()
                    rootroute = utils.getPathToRoot(parent, [])
                    feat = [connective, postag, ln, rn, lnpos, rnpos,
                            '-'.join(rootroute), refcon.sentencePosition]
                    enc_feat = [self.encode(v) for v in feat]
                    # taking first sent only in case ext arg is spread
                    # over multiple sentences
                    extargsent = list(set(t.sentenceId for t
                                          in rel.extArgTokens))[0]
                    sentposlabel = extargsent - refcon.sentenceId
                    X_train_pos.append(enc_feat)
                    y_train_pos.append(sentposlabel)
                    if sentposlabel == 0:
                        X_train_samesent.append(enc_feat)
                        if rel.extArgTokens[-1].tokenId < refcon.tokenId:
                            y_train_samesent.append(-1)
                        else:
                            y_train_samesent.append(1)

        self.sentposclf = RandomForestClassifier(class_weight='balanced',
                                                 n_estimators=1000,
                                                 random_state=42)
        self.samesentclf = RandomForestClassifier(class_weight='balanced',
                                                  n_estimators=1000,
                                                  random_state=42)

        self.sentposclf.fit(X_train_pos, y_train_pos)
        self.samesentclf.fit(X_train_samesent, y_train_samesent)

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        sys.stderr.write(
            'INFO: Done training explicit argument extractor'
            '...({:0>2}:{:0>2}:{:0>2})\n'.format(int(hours), int(minutes),
                                                 int(seconds)))

        # pickle.dump(self.sentposclf, codecs.open('sentposclf.pickle', 'wb'))
        # pickle.dump(self.samesentclf, codecs.open('samesentclf.pickle', 'wb')
        # )
        # sys.stderr.write('INFO: Saved classifier to sentposclf.pickle.\n')
        # sys.stderr.write('INFO: Saved classifier to samesentclf.pickle.\n')

    def load(self):

        if not os.path.exists(os.path.join(os.getcwd(), 'sentposclf.pickle')):
            return 'ERROR: sentposclf.pickle not found.\n'
        if not os.path.exists(os.path.join(os.getcwd(), 'samesentclf.pickle')):
            return 'ERROR: samesentclf.pickle not found.\n'

        self.sentposclf = pickle.load(codecs.open('sentposclf.pickle', 'rb'))
        self.samesentclf = pickle.load(codecs.open('samesentclf.pickle', 'rb'))

    def getGoldArgs(self, testfiles):

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
        goldrels = []
        for f in fd:
            pccTokens, relations = PCCParser.parseConnectorFile(
                fd[f]['connectives'])
            pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
            for rel in relations:
                if rel.relationType == 'explicit':
                    goldrels.append(rel)
        return goldrels

    def evaluate_gold(self, testfiles, f2gold):

        intarg_tp, intarg_fp, intarg_fn, extarg_tp, extarg_fp, extarg_fn =\
            0, 0, 0, 0, 0, 0
        for f in f2gold:
            if f in testfiles:
                relations, sents, tokens = f2gold[f]
                for rel in relations:
                    if rel.relationType == 'explicit':
                        refcon = rel.connective[0]
                        sentence = refcon.fullSentence
                        ptree = None
                        if sentence in self.parsermap:
                            ptree = ParentedTree.convert(
                                self.parsermap[sentence])
                        else:
                            sent = sentence.replace('(', '[')
                            sent = sent.replace(')', ']')
                            doc = self.nlp(sent)
                            ptree = Tree.fromstring(str(
                                doc.sentences[0].constituency))
                            self.parsermap[sentence] = ptree

                        tempextarg = self.getExtArg(ptree, rel, tokens, sents)
                        tempintarg = self.getIntArg(ptree, rel, tokens)
                        # since generally, intarg tokens are easier to
                        # predict than extarg tokens, filtering out any
                        # intarg tokens that may be in the extarg list
                        pred_intarg = tempintarg
                        pred_extarg = [x for x in tempextarg
                                       if x not in tempintarg]

                        gold_intarg = rel.arg2
                        gold_extarg = rel.arg1
                        # get tp fp fn here
                        gold_intarg = [x.tokenId for x in gold_intarg]
                        gold_extarg = [x.tokenId for x in gold_extarg]
                        for tid in set(gold_intarg + pred_intarg):
                            if tid in gold_intarg and tid in pred_intarg:
                                intarg_tp += 1
                            elif tid in gold_intarg:
                                intarg_fn += 1
                            elif tid in pred_intarg:
                                intarg_fp += 1
                        for tid in set(gold_extarg + pred_extarg):
                            if tid in gold_extarg and tid in pred_extarg:
                                extarg_tp += 1
                            elif tid in gold_extarg:
                                extarg_fn += 1
                            elif tid in pred_extarg:
                                extarg_fp += 1

        return intarg_tp, intarg_fp, intarg_fn, extarg_tp, extarg_fp, extarg_fn

    def evaluate_pred(self, pred_relations, gold_relations):

        intarg_tp = 0
        intarg_fp = 0
        intarg_fn = 0
        extarg_tp = 0
        extarg_fp = 0
        extarg_fn = 0
        for grel in gold_relations:
            grel_conn = sorted([int(x.tokenId) for x in grel.connectiveTokens])
            grel_intarg = [int(x.tokenId) for x in grel.intArgTokens]
            grel_extarg = [int(x.tokenId) for x in grel.extArgTokens]
            found = False
            for prel in pred_relations:
                prel_conn = sorted([x.tokenId for x in prel.connective])
                prel_intarg = [x.tokenId for x in prel.arg2]
                prel_extarg = [x.tokenId for x in prel.arg1]
                if prel_conn == grel_conn:
                    found = True
                    for tid in set(grel_intarg + prel_intarg):
                        if tid in grel_intarg and tid in prel_intarg:
                            intarg_tp += 1
                        elif tid in grel_intarg:
                            intarg_fn += 1
                        elif tid in prel_intarg:
                            intarg_fp += 1
                    for tid in set(grel_extarg + prel_extarg):
                        if tid in grel_extarg and tid in prel_extarg:
                            extarg_tp += 1
                        elif tid in grel_extarg:
                            extarg_fn += 1
                        elif tid in prel_extarg:
                            extarg_fp += 1
            if not found:  # count all tokens as false negatives
                for tid in grel_intarg:
                    intarg_fn += 1
                for tid in grel_extarg:
                    extarg_fn += 1

        return intarg_tp, intarg_fp, intarg_fn, extarg_tp, extarg_fp, extarg_fn

    def predict(self, relations, sents, tokens):

        for rel in relations:
            refcon = rel.connective[0]
            sentence = refcon.fullSentence
            sent = sentence.replace('(', '[')
            sent = sent.replace(')', ']')
            doc = self.nlp(sent)
            ptree = Tree.fromstring(str(
                doc.sentences[0].constituency))
            self.parsermap[sentence] = ptree
            ptree = ParentedTree.convert(ptree)

            tempextarg = self.getExtArg(ptree, rel, tokens, sents)
            tempintarg = self.getIntArg(ptree, rel, tokens)

            # since generally, intarg tokens are easier to predict than
            # extarg tokens, filtering out any intarg tokens that may be
            # in the extarg list
            intarg = tempintarg
            extarg = [x for x in tempextarg if x not in tempintarg]

            for iat in intarg:
                rel.addIntArgToken(tokens[iat])
            for eat in extarg:
                rel.addExtArgToken(tokens[eat])
