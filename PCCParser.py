# -*- coding: utf-8 -*-

import lxml.etree
import os
import sys
from collections import defaultdict

xmlp = lxml.etree.XMLParser(strip_cdata=False, resolve_entities=False,
                            encoding='utf-8', remove_comments=True)
syntaxTokenId = 1


class DiscourseToken:

    def __init__(self, tokenId, token):
        self.tokenId = tokenId
        self.token = token
        # setting to False by default, afterwards setting to True for
        # connectives
        self.isConnective = False

    def setStartIndex(self, val):
        self.characterStartIndex = val

    def setEndIndex(self, val):
        self.characterEndIndex = val

    def setSyntaxSentenceId(self, val):
        self.sentenceId = val

    def setFullSentence(self, val):
        self.fullSentence = val

    def setSentencePosition(self, val):
        self.sentencePosition = val

    def setPredictedConnective(self, val):
        self.predictedConnective = val

    def setMultiToken(self, y):  # this is only used for evaluation purposes
        if not hasattr(self, 'multiTokenIds'):
            self.multiTokenIds = []
        self.multiTokenIds.append(y)

    def setDocId(self, y):  # user for evaluation purposes only
        self.docId = y


class DiscourseRelation:

    def __init__(self, relationId):
        self.relationId = relationId
        self.connectiveTokens = []
        self.intArgTokens = []
        self.extArgTokens = []
        self.sense = None

    def setSense(self, sense):
        self.sense = sense

    def setType(self, _type):
        self.relationType = _type

    def addConnectiveToken(self, tid):
        self.connectiveTokens.append(tid)

    def addIntArgToken(self, tid):
        self.intArgTokens.append(tid)

    def addExtArgToken(self, tid):
        self.extArgTokens.append(tid)

    def filterIntArgForConnectiveTokens(self):
        self.intArgTokens = [x for x in self.intArgTokens
                             if x not in self.connectiveTokens]

    def setDocId(self, y):  # user for evaluation purposes only
        self.docId = y


def parseConnectorFile(cxml):

    tree = lxml.etree.parse(cxml, parser=xmlp)
    tokens = {}
    relations = []
    tokenoffset = 0
    for i, token in enumerate(tree.getroot().findall('.//token')):
        dt = DiscourseToken(token.get('id'), token.text)
        dt.setStartIndex(tokenoffset)
        dt.setEndIndex(tokenoffset + len(token.text))
        tokenoffset += len(token.text)+1
        tokens[int(token.get('id'))] = dt

    for relation in tree.getroot().findall('.//relation'):
        if relation.get('type') == 'explicit':
            dr = DiscourseRelation(relation.get('relation_id'))
            dr.setSense(relation.get('pdtb3_sense'))
            dr.setType(relation.get('type'))
            for cts in relation.findall('.//connective_tokens'):
                for ct in cts:
                    # not the case for implicit connectives
                    if 'id' in ct.attrib:
                        dr.addConnectiveToken(tokens[int(ct.get('id'))])
                        # setting token.isConnective boolean to True here
                        tokens[int(ct.get('id'))].isConnective = True
            for iat in relation.findall('.//int_arg_token'):
                dr.addIntArgToken(tokens[int(iat.get('id'))])
            for eat in relation.findall('.//ext_arg_token'):
                dr.addExtArgToken(tokens[int(eat.get('id'))])
            dr.filterIntArgForConnectiveTokens()
            dr.setDocId(os.path.splitext(os.path.basename(cxml))[0])
            relations.append(dr)

        elif relation.get('type') == 'implicit':
            dr = DiscourseRelation(relation.get('relation_id'))
            dr.setSense(relation.get('pdtb3_sense'))
            dr.setType(relation.get('type'))
            for iat in relation.findall('.//int_arg_token'):
                dr.addIntArgToken(tokens[int(iat.get('id'))])
            for eat in relation.findall('.//ext_arg_token'):
                dr.addExtArgToken(tokens[int(eat.get('id'))])
            dr.setDocId(os.path.splitext(os.path.basename(cxml))[0])
            relations.append(dr)

        # this is where this PCCParser should be continued if other
        # relation types (AltLex, EntRel, NoRel) should also be
        # included, i.e. elif relation.get('type') == 'AltLex':, etc.

    return tokens, relations


def parseSyntaxFile(sxml, tokens):

    global syntaxTokenId
    tree = lxml.etree.parse(sxml, parser=xmlp)
    sd = defaultdict(str)
    for body in tree.getroot():
        for elemid, sentence in enumerate(body):
            graph = sentence.getchildren()[0]
            terminalsNode = graph.find('.//terminals')
            nonterminalNodes = graph.find('.//nonterminals')
            tokenizedSentence = ' '.join([x.get('word') for x
                                          in terminalsNode])
            sd[elemid] = tokenizedSentence
            subdict, catdict = getSubDict(nonterminalNodes)
            for sentencePosition, t in enumerate(terminalsNode):
                sToken = t.get('word')
                dt = tokens[syntaxTokenId]
                dt.setSyntaxSentenceId(elemid)
                dt.setFullSentence(tokenizedSentence)
                dt.setSentencePosition(sentencePosition)
                if not sToken == dt.token:
                    sys.stderr.write(
                        'FATAL ERROR: Tokens do not match in %s: %s(%s) vs.'
                        ' %s(%s).\n' % (
                            sxml, sToken, str(syntaxTokenId),
                            tokens[syntaxTokenId].token,
                            str(tokens[syntaxTokenId].tokenId)))
                    sys.exit(1)

                syntaxTokenId += 1
    syntaxTokenId = 1  # reset at end of file
    return tokens


def getSubDict(nonterminalnodes):

    d = {}
    cat = {}
    for nt in nonterminalnodes:
        edges = []
        for edge in nt:
            # some have secedges, which cause nesting/loops
            if edge.tag == 'edge':
                edges.append(edge.get('idref'))
        d[nt.get('id')] = edges
        cat[nt.get('id')] = nt.get('cat')
    return d, cat


def wrapTokensInSentences(tokens):

    sid = defaultdict(list)
    for _id in tokens:
        sid[tokens[_id].sentenceId].append(tokens[_id])

    return sid
