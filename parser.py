# -*- coding: UTF-8 -*-

# __author__ = Huang Wenguan
# date  : 2017.3.15

from bs4 import BeautifulSoup
import jieba
import os
import os.path
import re


def parseHtmlfromFile(file):
    '''
    given a html, parse it, and return the text in it.
    at the first try, we only try to extract text in <span>
    itype : document(html) 
    rtype : str
    '''
    #parser = argparse.ArgumentParser(description="you know, just time")
    #parser.add_argument('--delta', default = 3)
    #args = parser.parse_args()

    res = []
    try:
        soup = BeautifulSoup(open(file), "lxml")
        y = soup.find_all('div')

        for i in y:
            res += list(i.stripped_strings)
    except:
        print('something wrong in ',str(file))
        res.append(str(file))
    return ''.join(res)


def jbTokenizer(raw_text):
    '''
    itype: str
    otype: list
    '''
    seg_list = jieba.cut(raw_text)
    return list(seg_list)

def sentTokenizer(raw_text):
    '''
    separate sentences first, which is encoded as a list of words, and return a list of sentences
    itype: str
    otype: list[list[str]]
    '''
    stop_punctuations = '[，。？（）《》]'
    sentences = re.split(stop_punctuations,raw_text)
    return [jbTokenizer(s) for s in sentences]

def purify(raw_list, tag = 'whole'):
    '''
    remove unecessary elements like
        1. number
        2. punctuation
    itype: list[str]
    otype: list[str]
    '''

    number_pattern = re.compile(r'.*[\d].*')
    containNumber = lambda x : number_pattern.match(x)
    punctuations = ',.()、－-%/／√： ' # ，。？（）《》 leave for separating sentences
    if tag == 'whole':
        return [x for x in raw_list if not containNumber(x) and not x in punctuations]
    elif tag == 'sent':
        return [purify(s) for s in raw_list]


def iterateFolder(htmlpath, outputpath):
    '''
    handle all files in htmlpath, an save the tokenized ouput in ouputpath
    '''
    print('starting iterating ...')
    for parent, dirnames, filenames in os.walk(htmlpath):
        print('total in ', str(len(filenames)))
        i = 0
        for filename in filenames:
            if 'html' not in filename: continue
            print('going for ', str(i))
            full_name = os.path.join(parent,filename)
            #print("parent is ", parent)
            #print("filename is ", filename)
            #print("the full name of the file is:", full_name)
            
            # real deal
            raw_text = parseHtmlfromFile(full_name)
            # this code will treat every file as a list of words
            #coarse_result = jbTokenizer(raw_text)   
            #fine_result = purify(coarse_result)  
            # this code will treat every file as a list of sentences, with each sentences as a list of words  
            sentences_separated_result = sentTokenizer(raw_text)
            fine_result = purify(sentences_separated_result, 'sent')

            # push it into a file
            output_filename = filename[:-4] + 'txt'
            output_fullname = os.path.join(outputpath, output_filename)
            output_file = open(output_fullname,'w')
            output_file.write(str(fine_result))
            output_file.close()
            i += 1


def main():
    htmlpath = '/home/vincent/tmp/shang1000/out'
    outputpath = '/home/vincent/tmp/out1000_sentence'
    iterateFolder(htmlpath, outputpath)


if __name__ == "__main__":
    main()