from argparse import ArgumentParser
import os
import pickle
import requests
import time
import multiprocessing
import math
from functools import partial
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

download_dir = '/media/m3rg2000/mounted/zaki/MatDB/hargun/downloads_springer/'
table_dir = '/media/m3rg2000/mounted/zaki/MatDB/crosserf_search/processed_springer/' 
 
def para_to_string(tag, soup, arefs = 1, normalize = 1):
    paper = ''
    if not normalize:
        return tag.text
    for obj in tag.contents:
        if obj.name == 'a' and arefs:
            if 'data-track-action' not in obj.attrs:
                paper += obj.text
                continue
            if obj['data-track-action'] == 'reference anchor':
                paper += '[BIB_REF]'
                paper += obj['title']
                paper += '[/BIB_REF]'
            if obj['data-track-action'] == 'figure anchor':
                fignum = obj['href'].split('#')[1]
                paper += '[FIG_REF]'
                paper += f'Figure {fignum[3:]}:'
                paper += para_to_string(soup.find(id= f'figure-{fignum[3:]}-desc').find('p'), soup, arefs = 0)
                paper += '[/FIG_REF]'  
            if obj['data-track-action'] == 'table anchor':
                tabnum = obj['href'].split('#')[1]
                paper += '[FIG_REF]'
                paper += f'Table {tabnum[3:]}:'
                paper += para_to_string(soup.find(id= tabnum), soup, arefs = 0)
                paper += '[/FIG_REF]'  
        elif obj.name == 'span' and obj['class'][0] == 'mathjax-tex':
            paper += '[TEX]'
            paper += obj.text
            paper += '[/TEX]'
        else :
            paper += obj.text
    return paper
          

def text_from_html(pii, journal, text_data, missed, normalize = 1):
    try:
        html_path = os.path.join(download_dir, journal, pii, f'{pii}.html')
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
        sections = soup.findAll('section')
        if not sections:
            missed.append(pii)
            return
        title = soup.find('meta', attrs={'name': 'dc.title'})['content']
        paper = f'Title: {title}. '
        stop = ['References', 'Bibliography', 'Acknowledgements', 'Appendix', 'CRediT', 'Data and Code']
        for section in sections:
            title = section.find(class_='c-article-section__title')
            content = section.find(class_='c-article-section__content')
            if not title:
                continue
            to_break = 0
            for stopper in stop:
                if stopper in title.text:
                    to_break = 1
                    break
            if to_break:
                break
            if ord(title.text[0])<57:
                paper += ' '.join(title.text.split(' ')[1:]) + ': '
            else:
                paper += title.text + ': '

            for tag in content.contents:
                if tag.name == 'p':
                    paper += para_to_string(tag, soup, 1, normalize = normalize)
        text_data[pii] = paper
        print(paper)
    except Exception as e:
        missed.append(pii)
        return

def process_journal(piis, journal):
    text_data = dict()
    missed = []
    for pii in piis:
        text_from_html(pii, journal, text_data, missed, normalize = 1) # normalize = 0 here if not normalizing
    return text_data, missed

journal_list = os.listdir(download_dir)

ncpus = 64
print("using", ncpus, "CPUs")
for journal in journal_list:
    if journal.startswith('.'):
        continue
    start_time = time.time()
    piis = os.listdir(os.path.join(download_dir, journal))
    if(len(piis)==0):
        continue
    chunk_size = math.ceil(len(piis)/ncpus)
    print(journal, len(piis))
    chunks = [piis[i:i+chunk_size] for i in range(0, len(piis), chunk_size)]
    partial_func = partial(process_journal, journal=journal)
    with multiprocessing.Pool(processes=ncpus) as pool:
        outputs = pool.map(partial_func, chunks)
    text_data = dict()
    missed = []
    print("At", time.time()-start_time, "seconds: parallel computation")
    for output in outputs:
        text_data.update(output[0])
        missed += output[1]
    print("At", time.time()-start_time, "seconds: merged outputs")
    print(len(missed), "documents missed")
    pickle.dump(text_data, open(os.path.join(table_dir, f'{journal}.pkl'), 'wb'))
    with open(os.path.join(table_dir, f'{journal}-missed.txt'), 'a') as f:
        for miss in missed:
            f.write(miss+'\n')
    print(time.time()-start_time, "seconds for", journal)