#!/usr/bin/env python3.8
import medspacy
from medspacy.section_detection import Sectionizer
from collections import namedtuple

nlp = medspacy.load(enable=[])
bwh_sectionizer = Sectionizer(nlp, rules='section_patterns.json')

Section = namedtuple("Section", ["category", "title", "body"])
def sectionize(document:str):
    '''
    document: EMR note

    Sectionize a document with medpacy and return a named tuple with
    category: medspacy's predefined categories 
    title: the string literal that matched medspacy's rules for section headers
    body: the body (text) of the section

    Returns a list of Section tuples to be used for corpus creation
    '''
    
    medspacy_sections = bwh_sectionizer(nlp(document))._.sections
    sections = [Section(sec.category, 
                sec.title_span, 
                sec.body_span) 
                for sec in medspacy_sections]
    return sections


def get_sections(input):
    '''
    input: list of EMR notes
    output: list of tuples (ih, ap) 
    '''
    ih = []
    ap = []
    for text in input:
        sections = sectionize(text)
        temp_ap = ''
        temp_internal = ''
        for section in sections:
            if section.category == 'History/Subjective':
                temp_internal+=str(section.body)
            if section.category == 'A/P':
                temp_ap+=str(section.body)
        if temp_internal:
            ih.append(temp_internal)
        else:
            ih.append('iNo_text')
        if temp_ap:
            ap.append(temp_ap)
        else:
            ap.append('aNo_text')
    return ih, ap


def delete_sections(input):
    '''
    input: list of EMR notes
    output: list of selected notes (keep all sections except header and none)   
    '''
    sec = []
    for text in input:
        sections = sectionize(text)
        temp_internal = ''
        for section in sections:
            if section.category != 'Header' and section.category != 'None':
                temp_internal+=str(section.body)
        if temp_internal:
            sec.append(temp_internal)
        else:
            sec.append('No_text')
    return sec