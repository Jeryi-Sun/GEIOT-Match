import json
data_extract_json = '../dataset/data_extract.json'
cails = []
case2id = {}
ids = 0
with open(data_extract_json, 'r') as f:
    for line in f:
        item = json.loads(line)
        case_a = "".join(item["case_A"][0])
        case_b = "".join(item["case_B"][0])
        if case_a in case2id:
            pass
        else:
            case2id[case_a] = "ecail"+str(ids)
            ids += 1
        if case_b in case2id:
            pass
        else:
            case2id[case_b] = "ecail"+str(ids)
            ids += 1
        item['id'] = case2id[case_a] + '|' + case2id[case_b]
        cails.append(item)

data_extract_json_new = '../dataset/data_extract_new.json'
with open(data_extract_json_new, 'w') as f:
    for item in cails:
        f.writelines(json.dumps(item, ensure_ascii=False))
        f.write('\n')
