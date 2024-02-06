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

download_dir = '/media/m3rg2000/mounted/zaki/MatDB/hargun/downloads/'
table_dir = '/media/m3rg2000/mounted/zaki/MatDB/crosserf_search/processed/'

def process(tag, refs, soup, normalize = 1):
    if not normalize:
        return tag.text.replace('\n', ' ')  
    if not tag.name:
        return str(tag).replace('\n', ' ')
    elif tag.name.startswith('cross-ref'):
        strr = ''
        if tag['refid'].startswith('BIB'):
            bibs = tag['refid'].split(' ')
            for bib in bibs:
                strr += '[BIB_REF]'
                if bib in refs:
                    strr += refs[bib]
                strr += '[/BIB_REF]'
        elif tag['refid'].startswith('TBL'):
            tbls = tag['refid'].split(' ')
            for tbl in tbls:
                strr += '[FIG_REF]'
                strr += f'Table {tbl[3:]}: {soup.find("table", {"id": tbl}).find("caption").find("simple-para").text}'
                strr += '[/FIG_REF]'
        elif tag['refid'].startswith('FIG'):
            figs = tag['refid'].split(' ')
            for fig in figs:
                strr += ' [FIG_REF]'
                strr += f'Figure {fig[3:]}: {soup.find("figure", {"id": fig}).find("caption").find("simple-para").text}'
                strr += '[/FIG_REF]'
        else:
            strr += tag.text.replace('\n', ' ')  
        return strr 
    else:
        strr = ''
        for content in tag.contents:
            strr += process(content, refs, soup, normalize = normalize)
        return strr    
 
def text_from_xml(pii, journal, text_data, missed, normalize = 1):
    try:
        xml_path = os.path.join(download_dir, journal, pii, f'{pii}.xml')

        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'xml')

        sec = soup.find('xocs:item-toc')
        if not sec:
            return
        en = sec.findAll('xocs:item-toc-entry')
        refs = {}
        bibs = soup.findAll('bib-reference')
        for bib in bibs:
            try:
                if bib.find('author'):
                    if bib.find('author').find('given-name'):
                        refs[bib['id']] = bib.find('maintitle').text + ', by ' + bib.find('author').find('given-name').text + bib.find('author').find('surname').text
                    else:
                        refs[bib['id']] = bib.find('maintitle').text + ', by ' + bib.find('author').find('surname').text
                elif bib.find('editor'):
                    if bib.find('editor').find('given-name'):
                        refs[bib['id']] = bib.find('maintitle').text + ', by ' + bib.find('editor').find('given-name').text + bib.find('editor').find('surname').text
                    else:
                        refs[bib['id']] = bib.find('maintitle').text + ', by ' + bib.find('editor').find('surname').text
                elif bib.find('textref'):
                    text = bib.find('textref').text
                    if ',' in text:
                        refs[bib['id']] = text.split(',')[1] + ' , by ' + text.split(',')[0]
                    else:
                        refs[bib['id']] = text
                else:
                    refs[bib['id']] = ''
            except:
                refs[bib['id']] = ''
        snum, sname = [], []
        for s in en:
            try:
                snum.append(s.find('xocs:item-toc-label').contents[0])
                sname.append(process(s.find('xocs:item-toc-section-title'), refs, soup, normalize = normalize))
            except:
                pass

        paper = {
            'Title': ' '.join(soup.find('dc:title').text.split(',')).strip() if soup.find('dc:title') else '',
            'Abstract': soup.find('dc:description').text.replace('Abstract', '').replace('\n', '').strip() if soup.find('dc:description') else '',
                }

        sname.insert(0, 'Abstract')
        snum.insert(0, '')
        all_sections = soup.find_all('ce:section')
    # sname contains all xocs:item-toc-label tag entries
    # all_sections contains all ce:section-title tag entries
    # may not be same
    # snum contains the actual section number, like 3.1, 3.1.1 etc.
    # the below code will create a dictionary of {major section number}_{section name}: {section content}
    # in each iteration all the text from one major section will be accumulated
        # print(sname, snum)
        cnt = 1
        for sec in all_sections:
            if not sec.find('section-title'):
                cnt += 1
                continue
            _sec = sec.find('section-title')
            if process(_sec, refs, soup, normalize = normalize) in sname:
                secid = cnt 
                if snum[secid].startswith('Appendix'):
                    break
                cnt += 1
                if '.' not in snum[secid]:
                    strr = ''
                    for tx in _sec.find_next_siblings():
                        strr += process(tx, refs, soup, normalize = normalize)
                    paper[f'{_sec.text}'] = strr.strip()

        output_str = ""
        for section in paper:
            if section=='Title':
                output_str += f'{section}: {paper[section]}'
                output_str += '. '
            else:
                output_str += f'{section}: {paper[section]} '
        output_str = ' '.join(output_str.replace('\t', ' ').split())
        
        text_data[pii] = output_str
        # print(output_str)
    except Exception as e:
        missed.append(pii)
        return

def process_journal(piis, journal):
    text_data = dict()
    missed = []
    for pii in piis:
        text_from_xml(pii, journal, text_data, missed, normalize = 1) # normalize = 0 for not normalizing
    return [text_data, missed]

journal_list = os.listdir(download_dir)
done_journals = os.listdir(table_dir)

ncpus = 64
print("using", ncpus, "CPUs")
for journal in journal_list:
    if journal.startswith('.'):
        continue
    start_time = time.time()
    piis = os.listdir(os.path.join(download_dir, journal))
    chunk_size = math.ceil(len(piis)/ncpus)
    print(chunk_size)
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
    pickle.dump(text_data, open(os.path.join(table_dir, f'{journal}.pkl'), 'wb'))
    with open(os.path.join(table_dir, f'{journal}-missed.txt'), 'a') as f:
        for miss in missed:
            f.write(miss+'\n')
    print(time.time()-start_time, "seconds for", journal)