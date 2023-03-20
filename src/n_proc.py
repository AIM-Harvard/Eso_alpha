import re
import torch

# Preprocesses the input text by performing various operations such as lowercasing, removing stop words, and regular expression substitutions.
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

# Fixes issues with text containing specific substrings, such as 'aNo_textiNo_text', 'iNo_text', and 'aNo_text'.
def text2sec(x):
    '''
    fix no ap and no ih issue
    'aNo_textiNo_text': sectionizer assement and plan section and interval history section not found
    'iNo_text': sectionizer interval history section not found
    'aNo_text': sectionizer assement and plan section not found
    '''
    if 'aNo_textiNo_text' in str(x['text']):
        return {'text': str(x['sec_text'])}
    elif 'iNo_text' in str(x['text']):
        return {'text': str(x['sec_text'])}
    elif 'aNo_text' in str(x['text']):
	    return {'text': str(x['sec_text'])}  
    else:
        return {'text': str(x['text'])}

# Converts the input value to binary labels (0 or 1) task 1.
def binary_preproc(x):
    if x == 0:
        return {'labels': 0}
    else:
        return {'labels': int(1)}

# Converts the input value to binary labels (0 or 1), task 2.
def bibi_preproc(x):
    if x < 2 :
        return {'labels': int(0)}
    else:
        return {'labels': int(1)}

# Converts the input value to ternary labels (0, 1, or 2) task 3.
def trinary_preproc(x):
    if x == 0:
        return {'labels': 0}
    elif x == 1:
        return {'labels': int(1)}
    else:
        return {'labels': int(2)}

# Returns the input value as a dictionary with the key 'degree'.
def grade2degree(x):
    return {'degree': x}

def grade_preproc(x):
    if x is None:
        return {'labels': 0}
    else:
        return {'labels': int(x)}

# Converts a grade value to a label (0 if None, otherwise the input grade).
def grade6_preproc(x):
    if x == 0:
        return {'labels': torch.tensor([1,0,1,0,0,0])}
    elif x == 1:
        return {'labels': torch.tensor([0,1,0,1,0,0])}
    elif x == 2:
        return {'labels': torch.tensor([0,1,0,0,1,0])}
    elif x == 3:
        return {'labels': torch.tensor([0,1,0,0,0,1])}

# Preprocesses structured text by concatenating specific keys and values.
def struc_text_preproc(x):
    temp = ''
    for i in ['TxDose', 'TxFx', 'esophv55', 'esophmean', 'technique']:
        temp += i
        temp += ':'
        temp += str(x[i])
        temp += ' '
    return {'Full Text': str(temp)+str(x['Full Text'])}

# Converts structured data to special tokens for input using specified thresholds.
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

# Returns an empty string as 'text' in a dictionary.
def project_text(x):
    return {'text': str('')}

# Concatenates 'ros' and 'text' fields in the input dictionary.
def text_ros(x):
    return {'text': str(x['ros'])+str(x['text'])}

# Concatenates 'exam' and 'text' fields in the input dictionary.
def text_exam(x):
    return {'text': str(x['exam'])+str(x['text'])}

# Concatenates 'rot' and 'text' fields in the input dictionary.
def text_rot(x):
    return {'text': str(x['rot'])+str(x['text'])}

# Concatenates 'ih' and 'text' fields in the input dictionary.
def text_ih(x):
    return {'text': str(x['ih'])+str(x['text'])}

# Concatenates 'ap' and 'text' fields in the input dictionary.
def text_ap(x):
    return {'text': str(x['ap'])+str(x['text'])}

# Returns the 'sec_text' field in the input dictionary.   
def text_sec(x):
    return {'text': str(x['sec_text'])}

# Similar to the text2sec function defined earlier but operates on a dictionary instead of a string.
def text2sec(x):
    '''
    fix no ap and no ih issue
    '''
    if 'aNo_textiNo_text' in str(x):
        return {'text': str(x['sec_text'])}
    else:
        return {'text': str(x['text'])} 

# Masks specific substrings in the input text using regular expressions.
def mask_struc(x):
    regex = r":  \d"
    subst = "**"
    result = re.sub(regex, subst, x, 0, re.MULTILINE)
    return {'text': str(result)}

# Similar to mask_struc, but for a different input format.
def mask_struc1(x):
    regex = r": \d"
    subst = "**"
    result = re.sub(regex, subst, str(x), 0, re.MULTILINE)
    return {'text': str(result)}

# Masks specific substrings in the input text by replacing '- None' with '**'.
def mask_struc2(x):
    regex = r"- None"
    subst = "**"
    result = re.sub(regex, subst, str(x), 0, re.MULTILINE)
    return {'text': str(result)}

# Removes the text after 'Toxicity Grading' in the input text.
def chunk_off1(x):
    try:
        index = x.index('Toxicity Grading')
        return {'text': str(x[:index])}
    except:
        index = len(x)
        return {'text': str(x[:index])}

# Removes the text after 'Assessment and Plan of Care' in the input text.
def chunk_off2(x):
    try:
        index = x.index('Assessment and Plan of Care')
        return {'text': str(x[:index])}
    except:
        index = len(x)
        return {'text': str(x[:index])}

# Returns the 'text' field in the input dictionary as 'Full Text'.
def text_full_text(x):
    return {'Full Text': str(x['text'])}