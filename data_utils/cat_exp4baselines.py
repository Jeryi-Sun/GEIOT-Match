import json
from copy import deepcopy
import copy
from tqdm import tqdm
data_type = 'CAIL'
bert_legal_all_sents_json = '../dataset/baselines_datasets/{}_bert_legal_all_sents.json'.format(data_type)
bert_legal_rationale_json = '../dataset/baselines_datasets/{}_bert_legal_rationale.json'.format(data_type)
bert_legal_wo_rationale_json = '../dataset/baselines_datasets/{}_bert_legal_wo_rationale.json'.format(data_type)
if data_type == 'CAIL':
    data_predictor_json = "/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/stage3/data_prediction_{}.json".format('graph')
elif data_type == 'ELAM':
    data_predictor_json = "/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/stage3/data_prediction_{}.json".format('graph')
else:
    exit()

data_predictor = []
with open(data_predictor_json, 'r') as f:
    for line in f:
        data_predictor.append(json.loads(line))

bert_legal_all_sents = []
with open(bert_legal_all_sents_json, 'r') as f:
    for line in f:
        bert_legal_all_sents.append(json.loads(line))

new_bert_legal_all_sents = []
for a in tqdm(bert_legal_all_sents):
    for b in data_predictor:
        if a['id'] == b['id']:
            c = copy.deepcopy(b)
            c['source_1_a'] = a['case_a']
            c['source_1_b'] = a['case_b']
            new_bert_legal_all_sents.append(c)
            break

bert_legal_rationale = []
with open(bert_legal_rationale_json, 'r') as f:
    for line in f:
        bert_legal_rationale.append(json.loads(line))
new_bert_legal_rationale = []
for a in tqdm(bert_legal_rationale):
    for b in data_predictor:
        if a['id'] == b['id']:
            c = copy.deepcopy(b)
            c['source_1_a'] = a['case_a']
            c['source_1_b'] = a['case_b']
            new_bert_legal_rationale.append(c)
            break

bert_legal_wo_rationale = []
with open(bert_legal_wo_rationale_json, 'r') as f:
    for line in f:
        bert_legal_wo_rationale.append(json.loads(line))

new_bert_legal_wo_rationale = []
for a in tqdm(bert_legal_wo_rationale):
    for b in data_predictor:
        if a['id'] == b['id']:
            c = copy.deepcopy(b)
            c['source_1_a'] = a['case_a']
            c['source_1_b'] = a['case_b']
            new_bert_legal_wo_rationale.append(c)
            break
with open(bert_legal_all_sents_json, 'w') as f:
    for line in new_bert_legal_all_sents:
        f.writelines(json.dumps(line, ensure_ascii=False))
        f.write('\n')

with open(bert_legal_rationale_json, 'w') as f:
    for line in new_bert_legal_rationale:
        f.writelines(json.dumps(line, ensure_ascii=False))
        f.write('\n')

with open(bert_legal_wo_rationale_json, 'w') as f:
    for line in new_bert_legal_wo_rationale:
        f.writelines(json.dumps(line, ensure_ascii=False))
        f.write('\n')