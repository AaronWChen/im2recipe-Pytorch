import json
import numpy as np
import sys
from tqdm import *
import time

def tok(text, ts=False):

    '''
    Usage: tokenized_text = tok(text,token_list)
    If token list is not provided default one will be used instead.
    
    I don't think this is tokenization, I think it's just a string joiner with a specific way to split words later...
    
    '''

    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']

    for t in ts:
        text = text.replace(t,' ' + t + ' ')

    return text

if __name__ == "__main__":
    '''
    Generate tokenized text for w2v training
    Words separated with ' '
    Different instructions separated with \t
    Different recipes separated with \n
    '''

    try:
        partition = str(sys.argv[1])
    except:
        partition = ''

    dets = json.load(open('../data/det_ingrs.json','r'))
    layer1 = json.load(open('../data/layer1.json','r'))

    idx2ind = {}
    ingrs = []
    recipe_count = 0
    for i,entry in enumerate(dets):
        idx2ind[entry['id']] = i


    t = time.time()
    print(f"Processing partition: {partition}")
    print(f"Saving tokenized instructions here: '../data/tokenized_instructions_{partition}.txt' ")
    print(f"Also using loop to count ingredients in {partition}")
    with open(f"../data/tokenized_instructions_{partition}.txt",'w+') as f:
        for i, entry in tqdm(enumerate(layer1)):
            # if partition is specified (train, test, val) and does not match the entry's partition, skip the loop
            if partition != '' and partition != entry['partition']:
                continue

            instrs = entry['instructions']

            allinstrs = ''
            for instr in instrs:
                instr =  instr['text']
                allinstrs += instr + '\t'

            # find corresponding set of detected ingredients
            det_ingrs = dets[idx2ind[entry['id']]]['ingredients']
            valid = dets[idx2ind[entry['id']]]['valid']

            for j, det_ingr in enumerate(det_ingrs):
                # if detected ingredient matches ingredient text,
                # means it did not work. We skip
                if not valid[j]:
                    continue
                
                # underscore ingredient
                det_ingr_undrs = det_ingr['text'].replace(' ','_')
                ingrs.append(det_ingr_undrs)
                allinstrs = allinstrs.replace(det_ingr['text'], det_ingr_undrs)

            # tokenize instructions with tok()
            allinstrs = tok(allinstrs)
            f.write(allinstrs + '\n')
            recipe_count += 1

    print(f"Processing time: {time.time() - t} seconds.")
    print(f"Number of unique ingredients in {partition}: {len(np.unique(ingrs))}")
    print(f"Number of unique recipes in {partition}: {recipe_count}")
