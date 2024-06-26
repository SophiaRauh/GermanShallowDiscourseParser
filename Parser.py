# -*- coding: utf-8 -*-

import argparse
import codecs
# import dill as pickle
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import configparser

import numpy
from flask import Flask
from flask import request
from flask_cors import CORS
from sklearn.metrics import f1_score, precision_score, recall_score
from spacy.lang.de import German
from tqdm import tqdm
# maybe necessary: peft, transformers for requirements

# custom modules
import ConnectiveClassifier
import ExplicitArgumentExtractor
import ExplicitSenseClassifier
import ImplicitSenseClassifier
import olha_viz
import PCCParser
import utils

nlp = German()
nlp.add_pipe('sentencizer')

app = Flask(__name__)
CORS(app)

cc = ConnectiveClassifier.ConnectiveClassifier()
eae = ExplicitArgumentExtractor.ExplicitArgumentExtractor()
esc = ExplicitSenseClassifier.ExplicitSenseClassifier()
isc = ImplicitSenseClassifier.ImplicitSenseClassifier()


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


def convert_reltypes(pcc_relations):

    parser_relations = []
    for rel in pcc_relations:
        nr = Relation(int(rel.relationId), rel.relationType, rel.docId)
        for t in rel.connectiveTokens:
            t.tokenId = int(t.tokenId)
            t.sentenceTokenId = t.sentencePosition
            nr.addConnectiveToken(t)
        for t in rel.intArgTokens:
            t.tokenId = int(t.tokenId)
            t.sentenceTokenId = t.sentencePosition
            nr.addIntArgToken(t)
        for t in rel.extArgTokens:
            t.tokenId = int(t.tokenId)
            t.sentenceTokenId = t.sentencePosition
            nr.addExtArgToken(t)
        nr.addSense(rel.sense)
        parser_relations.append(nr)
    return parser_relations


def custom_tokenize(inp):

    # using spacy sentencizer/tokenizer since most(all?) nltk ones
    # replace double quotes (and some other chars:
    # https://www.nltk.org/_modules/nltk/tokenize/treebank.html)
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


def relations2json(inp, relations):

    jso = []
    for rel in relations:
        js = {'ID': rel.relationId,
              'DocID': rel.docId,
              'Sense': rel.sense,
              'Type': rel.relationType}
        # Arg1
        arg1 = {}
        arg1_charspanlist = []
        arg1_rawtext = ''
        chunks = utils.get_chunks([x.tokenId for x in rel.arg1])
        for chu in chunks:  # can be discontinuous...
            spanlist = [
                [x.span[0] for x in rel.arg1 if x.tokenId == chu[0]][0],
                [x.span[1] for x in rel.arg1 if x.tokenId == chu[1]][0]]
            arg1_charspanlist.append(spanlist)
            arg1_rawtext += inp[spanlist[0]:spanlist[1]] + ' '
        arg1_rawtext = arg1_rawtext.strip()
        arg1_tokenlist = [[x.span[0], x.span[1], x.tokenId, x.sentenceId,
                           x.sentenceTokenId] for x in rel.arg1]
        arg1['CharacterSpanList'] = arg1_charspanlist
        arg1['RawText'] = arg1_rawtext
        arg1['TokenList'] = arg1_tokenlist
        js['Arg1'] = arg1

        # Arg2
        arg2 = {}
        arg2_charspanlist = []
        arg2_rawtext = ''
        chunks = utils.get_chunks([x.tokenId for x in rel.arg2])
        for chu in chunks:  # can be discontinuous...
            spanlist = [
                [x.span[0] for x in rel.arg2 if x.tokenId == chu[0]][0],
                [x.span[1] for x in rel.arg2 if x.tokenId == chu[1]][0]]
            arg2_charspanlist.append(spanlist)
            arg2_rawtext += inp[spanlist[0]:spanlist[1]] + ' '
        arg2_rawtext = arg2_rawtext.strip()
        arg2_tokenlist = [[x.span[0], x.span[1], x.tokenId, x.sentenceId,
                           x.sentenceTokenId] for x in rel.arg2]
        arg2['CharacterSpanList'] = arg2_charspanlist
        arg2['RawText'] = arg2_rawtext
        arg2['TokenList'] = arg2_tokenlist
        js['Arg2'] = arg2

        # Conn
        conn = {}
        conn_charspanlist = []
        conn_rawtext = ''
        chunks = utils.get_chunks([x.tokenId for x in rel.connective])
        for chu in chunks:
            spanlist = [
                [x.span[0] for x in rel.connective if x.tokenId == chu[0]][0],
                [x.span[1] for x in rel.connective if x.tokenId == chu[1]][0]]
            conn_charspanlist.append(spanlist)
            conn_rawtext += inp[spanlist[0]:spanlist[1]] + ' '
        conn_rawtext = conn_rawtext.strip()
        conn_tokenlist = [[x.span[0], x.span[1], x.tokenId, x.sentenceId,
                           x.sentenceTokenId] for x in rel.connective]
        conn['CharacterSpanList'] = conn_charspanlist
        conn['RawText'] = conn_rawtext
        conn['TokenList'] = conn_tokenlist
        js['Connective'] = conn

        jso.append(js)

    return json.dumps(jso, ensure_ascii=False)


def gold_eval(pcc_folder, files, numIterations, splits):

    sys.stderr.write('INFO: Starting gold evaluation...\n')
    connectivefiles = [x for x in utils.listfolder(
        os.path.join(pcc_folder, 'connectives'))]
    syntaxfiles = [x for x in utils.listfolder(
        os.path.join(pcc_folder, 'syntax'))]

    fd = defaultdict(lambda: defaultdict(str))
    fd = utils.addAnnotationLayerToDict(connectivefiles, fd, 'connectives')
    fd = utils.addAnnotationLayerToDict(syntaxfiles, fd, 'syntax')

    f2gold = {}
    for f in fd:
        pccTokens, relations = PCCParser.parseConnectorFile(
            fd[f]['connectives'])
        pccTokens = PCCParser.parseSyntaxFile(fd[f]['syntax'], pccTokens)
        sents = PCCParser.wrapTokensInSentences(pccTokens)
        relations = convert_reltypes(relations)
        f2gold[f] = (relations, sents, pccTokens)

    connective_fscores = []
    connective_pscores = []
    connective_rscores = []
    intarg_tp, intarg_fp, intarg_fn,\
        extarg_tp, extarg_fp, extarg_fn = 0, 0, 0, 0, 0, 0
    explicit_senses_detailed = []
    explicit_senses_second_level = []
    explicit_senses_first_level = []
    implicit_senses_detailed = []
    implicit_senses_second_level = []
    implicit_senses_first_level = []

    total = 0
    correct = 0
    for i in range(numIterations):
        sys.stderr.write('INFO: Starting iteration %s...\n' % str(i+1))
        testfiles = []
        trainfiles = []
        for index, _file in enumerate(files):
            if index >= splits[i] and index <= splits[i+1]:
                testfiles.append(_file)
            else:
                trainfiles.append(_file)
        cc.train(trainfiles)
        eae.train(trainfiles)
        esc.train(trainfiles)
        isc.train(trainfiles)

        # connectives
        pred, gold, f2tokens = cc.evaluate(testfiles)
        _cmf = f1_score(gold, pred, average='weighted')
        _cmp = precision_score(gold, pred, average='weighted')
        _cmr = recall_score(gold, pred, average='weighted')
        connective_fscores.append(_cmf)
        connective_pscores.append(_cmp)
        connective_rscores.append(_cmr)

        # explicit args
        i_tp, i_fp, i_fn, e_tp, e_fp, e_fn = eae.evaluate_gold(
            testfiles, f2gold)
        intarg_tp += i_tp
        intarg_fp += i_fp
        intarg_fn += i_fn
        extarg_tp += e_tp
        extarg_fp += e_fp
        extarg_fn += e_fn

        intarg_p, intarg_r, intarg_f = utils.getPrecisionRecallF1(
            intarg_tp, intarg_fp, intarg_fn)
        extarg_p, extarg_r, extarg_f = utils.getPrecisionRecallF1(
            extarg_tp, extarg_fp, extarg_fn)

        # explicit senses
        detailed_f1, second_level_f1,\
            first_level_f1, tot, cor = esc.evaluate_gold(testfiles, f2gold)

        total += tot
        correct += cor
        explicit_senses_detailed.append(detailed_f1)
        explicit_senses_second_level.append(second_level_f1)
        explicit_senses_first_level.append(first_level_f1)

        # implicit relations (senses)
        detailed_f1, second_level_f1, first_level_f1 = isc.evaluate_gold(
            testfiles, f2gold)
        implicit_senses_detailed.append(detailed_f1)
        implicit_senses_second_level.append(second_level_f1)
        implicit_senses_first_level.append(first_level_f1)

    print('connective precision:', numpy.mean(connective_pscores))
    print('connective recall:', numpy.mean(connective_rscores))
    print('connective f1:', numpy.mean(connective_fscores))
    intarg_p, intarg_r, intarg_f = utils.getPrecisionRecallF1(
        intarg_tp, intarg_fp, intarg_fn)
    extarg_p, extarg_r, extarg_f = utils.getPrecisionRecallF1(
        extarg_tp, extarg_fp, extarg_fn)
    print('intargprf:', intarg_p, intarg_r, intarg_f)
    print('extargprf:', extarg_p, extarg_r, extarg_f)

    print('Explicit sense full detail:', numpy.mean(explicit_senses_detailed))
    print('Explicit sense 2nd level:', numpy.mean(
        explicit_senses_second_level))
    print('Explicit sense 1st level:', numpy.mean(explicit_senses_first_level))

    print('Implicit sense full detail:', numpy.mean(implicit_senses_detailed))
    print('Implicit sense 2nd level:', numpy.mean(
        implicit_senses_second_level))
    print('Implicit sense 1st level:', numpy.mean(implicit_senses_first_level))


def evaluate():
    config = configparser.ConfigParser()
    config.read('config.ini')
    pcc_connectives_dir = os.path.join(config['PCC']['pccdir'], 'connectives')
    files = os.listdir(pcc_connectives_dir)
    numIterations = 2  # 10
    splits = utils.getDataSplits(numIterations, len(files))

    # gold_eval(pcc_folder, files, numIterations, splits)
    pred_eval(files, numIterations, splits)


def pred_eval(files, numIterations, splits):

    sys.stderr.write('INFO: Starting pred evaluation...\n')
    connective_fscores = []
    connective_pscores = []
    connective_rscores = []
    i_tp, i_fp, i_fn, e_tp, e_fp, e_fn = 0, 0, 0, 0, 0, 0
    explicit_senses_total = 0
    explicit_senses_detailed_correct = 0
    explicit_senses_second_level_correct = 0
    explicit_senses_first_level_correct = 0
    implicit_senses_total = 0
    implicit_senses_detailed_correct = 0
    implicit_senses_second_level_correct = 0
    implicit_senses_first_level_correct = 0

    for i in range(numIterations):
        sys.stderr.write('INFO: Starting iteration: %s\n' % str(i+1))
        testfiles = []
        trainfiles = []
        for index, _file in enumerate(files):
            if index >= splits[i] and index <= splits[i+1]:
                testfiles.append(_file)
            else:
                trainfiles.append(_file)
        cc.train(trainfiles)
        eae.train(trainfiles)
        esc.train(trainfiles)
        isc.train(trainfiles)

        pred, gold, f2tokens = cc.evaluate(testfiles)
        _cmf = f1_score(gold, pred, average='weighted')
        _cmp = precision_score(gold, pred, average='weighted')
        _cmr = recall_score(gold, pred, average='weighted')
        connective_fscores.append(_cmf)
        connective_pscores.append(_cmp)
        connective_rscores.append(_cmr)

        gold_arg_rels = eae.getGoldArgs(testfiles)
        gold_sense_rels = esc.getGoldSenses(testfiles)
        gold_implicits = isc.getGoldImplicits(testfiles)
        for f in tqdm(f2tokens):
            relations = []
            already_processed = []
            _rid = 1
            ftokens = f2tokens[f]
            tokens = {}
            for _id in ftokens:
                t = ftokens[_id]
                t.tokenId = int(t.tokenId)
                t.setDocId(f)
                t.sentenceTokenId = t.sentencePosition
                tokens[t.tokenId] = t
                if hasattr(t, 'predictedConnective') and not (
                        f, t.tokenId) in already_processed:
                    rel = Relation(_rid, 'Explicit', f)
                    rel.addConnectiveToken(t)
                    if hasattr(t, 'multiTokenIds'):
                        for ot in t.multiTokenIds:
                            ott = ftokens[int(ot)]
                            ott.tokenId = int(ott.tokenId)
                            t.sentenceTokenId = t.sentencePosition
                            rel.addConnectiveToken(ott)
                            already_processed.append((f, ott.tokenId))
                    relations.append(rel)
                    _rid += 1

            sents = PCCParser.wrapTokensInSentences(tokens)
            filegoldargs = [rel for rel in gold_arg_rels
                            if rel.docId == os.path.splitext(f)[0]]
            # by this time relations include connectives:
            eae.predict(relations, sents, tokens)
            intarg_tp, intarg_fp, intarg_fn,\
                extarg_tp, extarg_fp, extarg_fn = eae.evaluate_pred(
                    relations, filegoldargs)
            i_tp += intarg_tp
            i_fp += intarg_fp
            i_fn += intarg_fn
            e_tp += extarg_tp
            e_fp += extarg_fp
            e_fn += extarg_fn

            filegoldsenses = [rel for rel in gold_sense_rels
                              if rel.docId == os.path.splitext(f)[0]]
            # by this time relations also include args:
            esc.predict(relations)
            tot, dcor, scor, fcor = esc.evaluate_pred(relations,
                                                      filegoldsenses)

            explicit_senses_total += tot
            explicit_senses_detailed_correct += dcor
            explicit_senses_second_level_correct += scor
            explicit_senses_first_level_correct += fcor
            filegoldimplicits = [rel for rel in gold_implicits
                                 if rel.docId == os.path.splitext(f)[0]]
            newrels = isc.predict(relations, sents)
            maxrelid = max(
                [x.relationId for x in relations]) if relations else 0
            for nr in newrels:
                r = Relation(maxrelid+1, 'Implicit', f)
                maxrelid += 1
                for t in nr[0]:
                    r.addExtArgToken(t)
                for t in nr[1]:
                    r.addIntArgToken(t)
                r.addSense(nr[2])
                relations.append(r)
            itot, idcor, iscor, ifcor = isc.evaluate_pred(relations,
                                                          filegoldimplicits)
            implicit_senses_total += itot
            implicit_senses_detailed_correct += idcor
            implicit_senses_second_level_correct += iscor
            implicit_senses_first_level_correct += ifcor

    print('Pipeline results')
    print('=====================================')
    print('\tConnectives')
    print('\tprecision : %f' % numpy.mean(connective_pscores))
    print('\trecall    : %f' % numpy.mean(connective_rscores))
    print('\tf1        : %f' % numpy.mean(connective_fscores))
    print()
    intarg_p, intarg_r, intarg_f = utils.getPrecisionRecallF1(i_tp, i_fp, i_fn)
    extarg_p, extarg_r, extarg_f = utils.getPrecisionRecallF1(e_tp, e_fp, e_fn)
    print('=====================================')
    print('\tArguments')
    print('\tintarg precision : %f' % intarg_p)
    print('\tintarg recall    : %f' % intarg_r)
    print('\tintarg f1        : %f' % intarg_f)
    print()
    print('\textarg precision : %f' % extarg_p)
    print('\textarg recall    : %f' % extarg_r)
    print('\textarg f1        : %f' % extarg_f)
    print()
    print('=====================================')
    print('\tExplicit senses')
    print('\taccuracy third level  : %s' % str(
        float(explicit_senses_detailed_correct) / explicit_senses_total))
    print('\taccuracy second level : %s' % str(
        float(explicit_senses_second_level_correct) / explicit_senses_total))
    print('\taccuracy first level  : %s' % str(
        float(explicit_senses_first_level_correct) / explicit_senses_total))
    print()
    print('=====================================')
    print('\tImplicit senses')
    print('\taccuracy third level  : %s' % str(
        float(implicit_senses_detailed_correct) / implicit_senses_total))
    print('\taccuracy second level : %s' % str(
        float(implicit_senses_second_level_correct) / implicit_senses_total))
    print('\taccuracy first level  : %s' % str(
        float(implicit_senses_first_level_correct) / implicit_senses_total))
    print()


def test():

    start = time.time()
    cc.train()
    eae.train()
    esc.train()
    isc.train()

    testfolder = 'pcc2.2/tokenized/'
    for testfile in os.listdir(testfolder):
        atf = os.path.join(testfolder, testfile)
        sys.stderr.write('INFO: Processing file: %s\n' % atf)
        inp = codecs.open(atf).read()

        sents, tokens = custom_tokenize(inp)
        cc.predict(sents)

        # populating list of relations,
        # starting point are explicits/connectives
        relations = []
        _id = 1
        already_processed = []  # for phrasal connectives...
        for sid in sents:
            for i, token in enumerate(sents[sid]):
                if hasattr(token, 'isConnective')\
                        and token.tokenId not in already_processed:
                    rel = Relation(_id, 'Explicit', testfile)
                    rel.addConnectiveToken(token)
                    if hasattr(token, 'multiTokenIds'):
                        for ot in token.multiTokenIds:
                            rel.addConnectiveToken(tokens[ot])
                            already_processed.append(ot)
                    relations.append(rel)
                    _id += 1

        eae.predict(relations, sents, tokens)
        esc.predict(relations)

        newrels = isc.predict(relations, sents)
        maxrelid = max([x.relationId for x in relations]) if relations else 0
        for nr in newrels:
            r = Relation(maxrelid+1, 'Implicit', testfile)
            maxrelid += 1
            for t in nr[0]:
                r.addExtArgToken(t)
            for t in nr[1]:
                r.addIntArgToken(t)
            r.addSense(nr[2])
            relations.append(r)

        # to json output here
        jsonstr = relations2json(inp, relations)
        assert json.loads(jsonstr)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    sys.stderr.write(
        'INFO: Successfully parsed all files in test folder. Time taken:'
        ' {:0>2}:{:0>2}:{:0>2}\n.'.format(
            int(hours), int(minutes), int(seconds)))


def main(text_file):
    text_untokenized = []
    with open(text_file, 'r', encoding='utf-8') as file:
        for line in file:
            text_untokenized.append(line.strip())
    text_untokenized = ' '.join(text_untokenized)

    sents, tokens = custom_tokenize(text_untokenized)

    cc.train()
    eae.train()
    esc.train()
    isc.train()

    cc.predict(sents)

    # populating list of relations, starting point are explicits/connectives
    relations = []
    _id = 1
    already_processed = []  # for phrasal connectives...
    for sid in sents:
        for i, token in enumerate(sents[sid]):
            if hasattr(token, 'isConnective')\
                    and token.tokenId not in already_processed:
                rel = Relation(_id, 'Explicit', 'dummy')
                rel.addConnectiveToken(token)
                if hasattr(token, 'multiTokenIds'):
                    for ot in token.multiTokenIds:
                        rel.addConnectiveToken(tokens[ot])
                        already_processed.append(ot)
                relations.append(rel)
                _id += 1

    eae.predict(relations, sents, tokens)
    esc.predict(relations)

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

    # for dev/debugging, pickling result of the above
    # pickle.dump(sents, codecs.open('sents_debug.pickle', 'wb'))
    # pickle.dump(tokens, codecs.open('tokens_debug.pickle', 'wb'))
    # pickle.dump(relations, codecs.open('relations_debug.pickle', 'wb'))

    # sents = pickle.load(codecs.open('sents_debug.pickle', 'rb'))
    # tokens = pickle.load(codecs.open('tokens_debug.pickle', 'rb'))
    # relations = pickle.load(codecs.open('relations_debug.pickle', 'rb'))

    jsonstr = relations2json(text_untokenized, relations)

    with open(f"{Path(text_file).stem}_results.json", "w",
              encoding="utf-8") as file:
        json.dump(json.loads(jsonstr), file, indent=4,
                  sort_keys=True, ensure_ascii=False)


# function to load trained&pickled components
@app.route('/load', methods=['GET'])
def load():

    start = time.time()
    cc.load()
    eae.load()
    esc.load()
    isc.load()
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    return 'INFO: Successfully loaded pre-trained components. Time taken:'\
        ' {:0>2}:{:0>2}:{:0>2}\n.'.format(int(hours), int(minutes),
                                          int(seconds))


@app.route('/train', methods=['GET'])
def train():

    start = time.time()

    cc.train()
    eae.train()
    esc.train()
    isc.train()

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    return 'INFO: Successfully trained models. Time taken:'\
        ' {:0>2}:{:0>2}:{:0>2}\n.'.format(
            int(hours), int(minutes), int(seconds))


@app.route('/parse', methods=['GET', 'POST'])
def parse():

    if not hasattr(cc, 'clfs') or not hasattr(eae, 'sentposclf')\
            or not hasattr(esc, 'clfs') or not hasattr(isc, 'mlp'):
        return 'ERROR: Could not find trained models. Are you sure you used'\
            ' the train endpoint?\n'

    inp = None
    if request.args.get('input'):
        inp = request.args.get('input')
    elif request.files['input']:
        inp = request.files['input'].read().decode('utf-8')
    else:
        return 'INFO: Please provide input text.\n'

    docid = None
    if request.args.get('docid') is None:
        docid = time.strftime('%m-%d-%Y_%H:%M:%S', time.gmtime())
    else:
        docid = request.args.get('docid')

    sents, tokens = custom_tokenize(inp)
    cc.predict(sents)

    # populating list of relations, starting point are
    # explicits/connectives
    relations = []
    _id = 1
    already_processed = []  # for phrasal connectives...
    for sid in sents:
        for i, token in enumerate(sents[sid]):
            if hasattr(token, 'isConnective')\
                    and token.tokenId not in already_processed:
                rel = Relation(_id, 'Explicit', docid)
                rel.addConnectiveToken(token)
                if hasattr(token, 'multiTokenIds'):
                    for ot in token.multiTokenIds:
                        rel.addConnectiveToken(tokens[ot])
                        already_processed.append(ot)
                relations.append(rel)
                _id += 1

    eae.predict(relations, sents, tokens)
    esc.predict(relations)

    newrels = isc.predict(relations, sents)
    maxrelid = max([x.relationId for x in relations]) if relations else 0
    for nr in newrels:
        r = Relation(maxrelid+1, 'Implicit', docid)
        maxrelid += 1
        for t in nr[0]:
            r.addExtArgToken(t)
        for t in nr[1]:
            r.addIntArgToken(t)
        r.addSense(nr[2])
        relations.append(r)

    # to json output here
    jsonstr = relations2json(inp, relations)

    # including option to dump output in brat format required for
    # Olha Zolotarenko's visualisation component (see olha_viz.py)
    if request.args.get('viz'):
        jsonstr = olha_viz.convert(inp, jsonstr)

    return jsonstr


if __name__ == '__main__':

    # main()  # for running without flask
    # test()
    # evaluate()
    # sys.exit()

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--port',
    #                        help='port number to start flask app on',
    #                        default=5000, type=int)
    argparser.add_argument(
        '-c', '--corpus', action='store', type=str,
        help='Path to a text corpus for connective classification')
    args = argparser.parse_args()
    if args.corpus:
        main(args.corpus)
    else:
        evaluate()

    # app.run(host='0.0.0.0', port=args.port)
