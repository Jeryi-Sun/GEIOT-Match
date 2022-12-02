import json

data_type = "CAIL"
if data_type == "CAIL":
    dismatch_files = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/dismatch_data_prediction_t5_graph.json"

    midmatch_files = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/midmatch_data_prediction_t5_graph.json"

    match_files = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/match_data_prediction_t5_graph.json"

    GE_file = "/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/stage3/data_prediction_graph.json"

    ELAM_save_dataset_rationale = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/rationale_CAIL_dataset.json"
else:
    dismatch_files = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/ELAM/dismatch_data_prediction_t5_graph.json"

    midmatch_files = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/ELAM/midmatch_data_prediction_t5_graph.json"

    match_files = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/ELAM/match_data_prediction_t5_graph.json"

    GE_file = "/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/stage3/data_prediction_graph.json"

    ELAM_save_dataset_rationale = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/rationale_ELAM_dataset.json"

GE_data = []
ELAM_rationale_data = []
with open(ELAM_save_dataset_rationale, 'r') as f:
    for line in f:
        item = json.loads(line)
        ELAM_rationale_data.append(item)

with open(match_files, 'r') as f:
    for line in f:
        item = json.loads(line)
        source_id = item['id']
        for item_b in ELAM_rationale_data:
            target_id = item_b['id']
            if source_id == target_id:
                item.update(item_b)
        GE_data.append(item)

with open(midmatch_files, 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)

        GE_data[i]['exp'] += item['exp']

with open(dismatch_files, 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        GE_data[i]['exp'] += item['exp']

with open(GE_file, 'w') as f:
    for item in GE_data:
        f.writelines(json.dumps(item, ensure_ascii=False))
        f.write('\n')



