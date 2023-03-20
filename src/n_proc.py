import re
import torch


def text_preproc(x, lower=False, stop_words=False, punctuation=False):
    if lower:
        x = x.lower()
    if stop_words:
        x = ' '.join([word for word in x.split(' ') if word not in stop_words])
    x = x.encode('ascii', 'ignore').decode()
    x = re.sub(r'https*\S+', ' ', x)
    x = re.sub(r'@\S+', ' ', x)
    x = re.sub(r'#\S+', ' ', x)
    x = re.sub(r'\s{2,}', ' ', x)
    x = re.sub(r'[G-g]rade\s\d', '', x)
    return {'clean_text': x}

def text2sec(x):
    '''
    fix no ap and no ih issue
    '''
    if 'aNo_textiNo_text' in str(x['text']):
        return {'text': str(x['sec_text'])}
    elif 'iNo_text' in str(x['text']):
        return {'text': str(x['sec_text'])}
    elif 'aNo_text' in str(x['text']):
	    return {'text': str(x['sec_text'])}  
    else:
        return {'text': str(x['text'])}

def bibi_preproc(x):
    if x < 2 :
        return {'labels': int(0)}
    else:
        return {'labels': int(1)}

def binary_preproc(x):
    if x == 0:
        return {'labels': 0}
    else:
        return {'labels': int(1)}

def trinary_preproc(x):
    if x == 0:
        return {'labels': 0}
    elif x == 1:
        return {'labels': int(1)}
    else:
        return {'labels': int(2)}

def grade2degree(x):
    return {'degree': x}

def grade_preproc(x):
    if x is None:
        return {'labels': 0}
    else:
        return {'labels': int(x)}

def grade6_preproc(x):
    if x == 0:
        return {'labels': torch.tensor([1,0,1,0,0,0])}
    elif x == 1:
        return {'labels': torch.tensor([0,1,0,1,0,0])}
    elif x == 2:
        return {'labels': torch.tensor([0,1,0,0,1,0])}
    elif x == 3:
        return {'labels': torch.tensor([0,1,0,0,0,1])}


def struc_text_preproc(x):
    temp = ''
    for i in ['TxDose', 'TxFx', 'esophv55', 'esophmean', 'technique']:
        temp += i
        temp += ':'
        temp += str(x[i])
        temp += ' '
    return {'Full Text': str(temp)+str(x['Full Text'])}

def struc_text_preproc1(x):
    """
    Use threshholds to convert structured data to special tokens for input
    TODO: 1. maybe have even finer granularity? I.e., more than binary threshhold 
          2. possibly make thresholds as tunable arguments? 
    """
    # txD_thresh = None ['<esomean_high>', '<esomean_low>',‘3dcrt’, ‘<UNK>’, ‘imrt’, ‘imrt/3dcrt’, ‘vmat’]
    # txf_thresh = None
    esomean_thresh = 34
    structured_input = ['<esomean_high>', x['technique'], '[SEP]', str(x['Full Text'])]
    # if x['TxDose'] < txD_thresh:
    #     structured_input[0] =  '<txd_low>'
    # if x['TxFx'] < txf_thresh:
    #     structured_input[1] = '<txf_low>'
    if x['esophmean'] < esomean_thresh:
        structured_input[0] = '<esomean_low>'
    return {'Full Text': ' '.join(structured_input)}


def project_text(x):
    return {'text': str('')}

def text_ros(x):
    return {'text': str(x['ros'])+str(x['text'])}

def text_exam(x):
    return {'text': str(x['exam'])+str(x['text'])}

def text_rot(x):
    return {'text': str(x['rot'])+str(x['text'])}

def text_ih(x):
    return {'text': str(x['ih'])+str(x['text'])}

def text_ap(x):
    return {'text': str(x['ap'])+str(x['text'])}
    
def text_sec(x):
    return {'text': str(x['sec_text'])}

def text2sec(x):
    '''
    fix no ap and no ih issue
    '''
    if 'aNo_textiNo_text' in str(x):
        return {'text': str(x['sec_text'])}
    else:
        return {'text': str(x['text'])} 

def mask_struc(x):
    regex = r":  \d"
    subst = "**"
    result = re.sub(regex, subst, x, 0, re.MULTILINE)
    return {'text': str(result)}

def mask_struc1(x):
    regex = r": \d"
    subst = "**"
    result = re.sub(regex, subst, str(x), 0, re.MULTILINE)
    return {'text': str(result)}

def mask_struc2(x):
    regex = r"- None"
    subst = "**"
    result = re.sub(regex, subst, str(x), 0, re.MULTILINE)
    return {'text': str(result)}

def chunk_off1(x):
    try:
        index = x.index('Toxicity Grading')
        return {'text': str(x[:index])}
    except:
        index = len(x)
        return {'text': str(x[:index])}

def chunk_off2(x):
    try:
        index = x.index('Assessment and Plan of Care')
        return {'text': str(x[:index])}
    except:
        index = len(x)
        return {'text': str(x[:index])}

def text_full_text(x):
    return {'Full Text': str(x['text'])}