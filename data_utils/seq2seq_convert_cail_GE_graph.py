import sys
sys.path.append("..")
from models.selector_two_multi_class_ot_cail_v2 import Selector2_mul_class, args, load_checkpoint, OT, load_data, data_extract_npy, data_extract_json, device
import torch
from utils.snippets import *
import torch
import json
from tqdm import tqdm
"""
todo 需要添加一个分类器，对相似句子与不相似句子进行分类
"""
def model_class(model, OT_model, case_A, case_B, seq_len_A, seq_len_B):
    """
    :param model:
    :param OT_model:
    :param case_A:
    :param case_B:
    :return: AO, YO, ZO, AI, YI, ZI
    """
    O_a, I_a = [], []
    O_b, I_b = [], []
    output_batch_A, batch_mask_A = model(case_A)
    output_batch_B, batch_mask_B = model(case_B)
    plan_list = OT_model(output_batch_A, output_batch_B, case_A, case_B, None,
                            batch_mask_A, batch_mask_B, model_type="valid")
    OT_matrix = torch.ge(plan_list, 1 / case_A.shape[1] / args.threshold_ot).long()
    vec_correct_A = torch.argmax(output_batch_A, dim=-1).long()[0][:seq_len_A]
    vec_correct_B = torch.argmax(output_batch_B, dim=-1).long()[0][:seq_len_B]
    relation_A = torch.sum(OT_matrix[0], dim=1)
    relation_B = torch.sum(OT_matrix[0], dim=0)

    for i, label in enumerate(vec_correct_A):
        if label == 1:
            if relation_A[i] >= 1:
                I_a.append(i)
            else:
                O_a.append(i)

    for i, label in enumerate(vec_correct_B):
        if label == 1:
            if relation_B[i] >= 1:
                I_b.append(i)
            else:
                O_b.append(i)
    pair_sentences = []
    for a in range(seq_len_A):
        for b in range(seq_len_B):
            if OT_matrix[0][a][b]:
                if vec_correct_A[a] and vec_correct_B[b]:
                    pair_sentences.append([a, b])
    O, I = [O_a, O_b], [I_a, I_b]

    return O, I, [vec_correct_A+(torch.ge(relation_A[:seq_len_A], 1)*1)*vec_correct_A, vec_correct_B+(torch.ge(relation_B[:seq_len_B], 1)*1)*vec_correct_B], pair_sentences




def generate_text_cluster(case_a, case_b, d, O, I, all_true,  I_true):
    source_1_a = ''.join(["[O]" + case_a[0][i] for i in O[0]] + ["[I]" + case_a[0][i] for i in I[0]])

    source_1_b = ''.join(["[O]" + case_b[0][i] for i in O[1]] + ["[I]" + case_b[0][i] for i in I[1]])

    source_2_a = ''.join(["[O]" + case_a[0][i] for i in all_true[0] if i not in I_true[0]] + ["[I]" + case_a[0][i] for i in I_true[0]])

    source_2_b = ''.join(["[O]" + case_b[0][i] for i in all_true[1] if i not in I_true[1]] + ["[I]" + case_b[0][i] for i in I_true[1]])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': d['explanation'],
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result


def get_extract_text(case_a, prediction):
    source_1_a = ''
    for i, output_class in enumerate(prediction):
        if output_class == 1:
            source_1_a += "[O]" + case_a[0][i]
        elif output_class == 2:
            source_1_a += "[I]" + case_a[0][i]
        else:
            pass
    return source_1_a


def get_extract_text_wo_token(case_a, prediction):
    source_1_a = ''
    for i, output_class in enumerate(prediction):
        if output_class != 0:
            source_1_a += case_a[0][i]
        else:
            pass
    return source_1_a


def generate_text_sort(case_a, case_b, d, prediction, label):

    source_1_a = get_extract_text(case_a, prediction[0])
    source_1_b = get_extract_text(case_b, prediction[1])
    source_2_a = get_extract_text(case_a, label[0])
    source_2_b = get_extract_text(case_b, label[1])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': d['explanation'],
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result


def generate_text_wo_token(case_a, case_b, d, prediction, label):

    source_1_a = get_extract_text_wo_token(case_a, prediction[0])
    source_1_b = get_extract_text_wo_token(case_b, prediction[1])
    source_2_a = get_extract_text_wo_token(case_a, label[0])
    source_2_b = get_extract_text_wo_token(case_b, label[1])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': d['explanation'],
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result

import re
def extract_text_law(text):
    all_law = []
    pattern = re.compile(u'《.*?》|第[零|一|二|三|四|五|六|七|八|九|十|个|十|百|千]{1,10}?条')
    m = re.findall(pattern=pattern, string=text)
    """
    过滤一下，《》内必须有法，否则删除并且将后续的不带《》的都删除
    """
    i = 0
    while i < len(m):
        if m[i][0] == '《' and "法" in m[i]:
            temp_idx = i
            i += 1
            while i < len(m) and m[i][0] != '《':  # 去除一下非法律的书名号内容
                all_law.append(m[temp_idx] + m[i])
                i += 1
        elif m[i][0] == '《' and "法" not in m[i]:
            i += 1
            while i < len(m) and re.search(r"《.*》", m[i]) == None:
                i += 1
        else:
            print(m[i])
            i += 1
            continue
    a = r'《(.*?)》'
    xingfa_laws = []
    for law_content in all_law:
        law = re.findall(a, law_content)[0]
        if law == "中华人民共和国民法":
            xingfa_laws.append(law_content) # 其实是民法
    return xingfa_laws
minfa_dic = "/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data/minfa_law_dic.json"
def get_law_text(text):
    tiao_text_A = ""
    a = r'《(.*?)》'
    with open(minfa_dic, 'r') as f:
        xingfalaws = json.load(f)
    law = re.findall(a, text)[0]
    if law == "中华人民共和国民法":
        tiao = text.split("》")[1]
        if tiao in xingfalaws.keys():
            tiao_text_A = xingfalaws[tiao]
    return tiao_text_A

def generate_text_GE_graph(case_a, case_b, d, O, I, all_true, I_true, prediction_relation, label_realtion):

    def generate_source_a_b(relation, I):
        inter_pairs = []
        source_1_a_b = ""
        for item in relation:
            a, b = item
            if a in I[0]:
                source_1_a_b += '[I]'+case_a[0][a]
            else:
                print("error at generate_text_GE_graph")
            source_1_a_b += '[R]'

            if b in I[1]:
                source_1_a_b += '[I]'+case_b[0][b]
            else:
                print("error at generate_text_GE_graph")
            inter_pairs.append([a, b])
        return source_1_a_b, inter_pairs

    law_a = extract_text_law("".join(case_a[0]))
    law_b = extract_text_law("".join(case_b[0]))
    source_a_b_law = ""
    for l in law_a:
        if l in law_b:
            source_a_b_law += "[LI]" + get_law_text(l)
        else:
            source_a_b_law += "[LO]" + get_law_text(l)
    for l in law_b:
        if l in law_a:
            pass
        else:
            source_a_b_law += "[LO]" + get_law_text(l)
    source_1_a_b, inter_pairs = generate_source_a_b(prediction_relation, I)
    source_1_a = ''.join(["[O]" + case_a[0][i] for i in O[0]])

    source_1_b = ''.join(["[O]" + case_b[0][i] for i in O[1]])

    source_2_a_b, _ = generate_source_a_b(label_realtion, I_true)
    source_2_a = ''.join(
        ["[O]" + case_a[0][i] for i in all_true[0] if i not in I_true[0]])

    source_2_b = ''.join(
        ["[O]" + case_b[0][i] for i in all_true[1] if i not in I_true[1]])

    result = {
        'source_1': source_1_a_b + source_1_a + source_1_b,
        'source_2': source_2_a_b + source_2_a + source_2_b,
        'explanation': d['explanation'],
        'source_1_dis': [source_1_a, source_1_b, source_1_a_b, source_a_b_law],
        'source_2_dis': [source_2_a, source_2_b, source_2_a_b, source_a_b_law],
        'label': d['label'],
        "id":d['id']
    }
    return result, inter_pairs


def fold_convert_cail_ot(data, data_x, type, generate=False, generate_mode = 'cluster'):
    """每一fold用对应的模型做数据转换
    """

    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent))
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract_ot-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent))
        results = []
        print(type+"ing")
        for i, d in enumerate(data):
            if type == 'match' and d["label"] == 2 or type == 'midmatch' and d["label"] == 1 or type == 'dismatch' and d["label"] == 0:
                case_a = d['case_A']
                case_b = d['case_B']
                important_A, important_B = [], []
                data_y_seven_class_A, data_y_seven_class_B = [0]*len(case_a[0]), [0]*len(case_b[0])
                for pos in d['relation_label']:
                    row, col = pos[0], pos[-1]
                    important_A.append(row)
                    important_B.append(col)

                for j in case_a[1]:
                    if j[0] in important_A:
                        data_y_seven_class_A[j[0]] = j[1] + 1
                    else:
                        data_y_seven_class_A[j[0]] = j[1]

                for j in case_b[1]:
                    if j[0] in important_B:
                        data_y_seven_class_B[j[0]] = j[1] + 1
                    else:
                        data_y_seven_class_B[j[0]] = j[1]
                label = [data_y_seven_class_A, data_y_seven_class_B]



                O, I, prediction, pair_sentence = model_class(model, ot_model, torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device),
                                                     torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device),len(case_a[0]), len(case_b[0]))

                all_true, I_true = [], []
                label_pair_sentence = d['relation_label']
                temp_a, temp_b = [], []
                for i in d['relation_label']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                I_true.append(temp_a)
                I_true.append(temp_b)


                case_a_temp = []
                for i in case_a[1]:
                    if i[1] == 1:
                        case_a_temp.append(i[0])
                all_true.append(case_a_temp)

                case_b_temp = []
                for i in case_b[1]:
                    if i[1] == 1:
                        case_b_temp.append(i[0])

                all_true.append(case_b_temp)
                if generate:
                    if generate_mode == 'cluster':
                        results.append(generate_text_cluster(case_a, case_b, d, O, I, all_true, I_true))
                    elif generate_mode == 'sort':
                        results.append(generate_text_sort(case_a, case_b, d, prediction, label))
                    elif generate_mode == 'graph':
                        result, inter_pairs = generate_text_GE_graph(case_a, case_b, d, O, I, all_true, I_true, pair_sentence, label_pair_sentence)
                        results.append(result)
                    else:
                        results.append(generate_text_wo_token(case_a, case_b, d, prediction, label))

        if generate:
            return results

def generate_rationale_text(case_a, case_b, d, prediction, label):

    source_1_a = get_extract_text_wo_token(case_a, prediction[0])
    source_1_b = get_extract_text_wo_token(case_b, prediction[1])
    source_2_a = get_extract_text_wo_token(case_a, label[0])
    source_2_b = get_extract_text_wo_token(case_b, label[1])

    result = {
        "id": d['id'],
        'source_1_a': source_1_a,
        'source_1_b': source_1_b,
        'source_2_a': source_2_a,
        'source_2_b': source_2_b
    }
    return result


def generate_rationale_func(data, data_x):
    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent))
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract_ot-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent))
        results = []
        for i, d in tqdm(enumerate(data)):
            case_a = d['case_A']
            case_b = d['case_B']
            important_A, important_B = [], []
            data_y_seven_class_A, data_y_seven_class_B = [0]*len(case_a[0]), [0]*len(case_b[0])
            for pos in d['relation_label']:
                row, col = pos[0], pos[-1]
                important_A.append(row)
                important_B.append(col)

            for j in case_a[1]:
                if j[0] in important_A:
                    data_y_seven_class_A[j[0]] = j[1] + 1
                else:
                    data_y_seven_class_A[j[0]] = j[1]

            for j in case_b[1]:
                if j[0] in important_B:
                    data_y_seven_class_B[j[0]] = j[1] + 1
                else:
                    data_y_seven_class_B[j[0]] = j[1]
            label = [data_y_seven_class_A, data_y_seven_class_B]



            O, I, prediction, pair_sentence = model_class(model, ot_model, torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device),
                                                 torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device),len(case_a[0]), len(case_b[0]))


            results.append(generate_rationale_text(case_a, case_b, d, prediction, label))

        return results


def convert(filename, data, data_x, type,  generate_mode):
    """转换为生成式数据
    """
    total_results = fold_convert_cail_ot(data, data_x, type, generate=True, generate_mode=generate_mode)

    with open(filename, 'w') as f:
        for item in total_results:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')


def fold_convert_our_data_graph(data, data_x):
    """每一fold用对应的模型做数据转换
    """

    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent))
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract_ot-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent))
        for i, d in enumerate(data):
            case_a = d['case_A']
            case_b = d['case_B']
            important_A, important_B = [], []
            data_y_seven_class_A, data_y_seven_class_B = [0] * len(case_a[0]), [0] * len(case_b[0])
            for pos in d['relation_label']:
                row, col = pos[0], pos[-1]
                important_A.append(row)
                important_B.append(col)

            for j in case_a[1]:
                if j[0] in important_A:
                    data_y_seven_class_A[j[0]] = j[1] + 1
                else:
                    data_y_seven_class_A[j[0]] = j[1]

            for j in case_b[1]:
                if j[0] in important_B:
                    data_y_seven_class_B[j[0]] = j[1] + 1
                else:
                    data_y_seven_class_B[j[0]] = j[1]
            label = [data_y_seven_class_A, data_y_seven_class_B]

            O, I, prediction, pair_sentence = model_class(model, ot_model,
                                                          torch.tensor(np.expand_dims(data_x[2 * i], axis=0),
                                                                       device=device),
                                                          torch.tensor(np.expand_dims(data_x[2 * i + 1], axis=0),
                                                                       device=device), len(case_a[0]), len(case_b[0]))

            all_true, I_true = [], []

            temp_a, temp_b = [], []
            for i in d['relation_label']:
                temp_a.append(i[0])
                temp_b.append(i[1])
            I_true.append(temp_a)
            I_true.append(temp_b)

            case_a_temp = []
            for i in case_a[1]:
                if i[1] == 1:
                    case_a_temp.append(i[0])
            all_true.append(case_a_temp)

            case_b_temp = []
            for i in case_b[1]:
                if i[1] == 1:
                    case_b_temp.append(i[0])

            all_true.append(case_b_temp)
            label_pair_sentence = d['relation_label']
            _, inter_pairs = generate_text_GE_graph(case_a, case_b, d, O, I, all_true, I_true, pair_sentence, label_pair_sentence)

            d["inter_pairs"] = inter_pairs
            d["rationale_a"] = O[0]+I[0]
            d["rationale_b"] = O[1]+I[1]
            for item in d["inter_pairs"]:
                if item[0] not in d["rationale_a"] or item[1] not in d["rationale_b"]:
                        print(1)


if __name__ == '__main__':

    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    make_graph_data = False
    generate_rationale = True
    if make_graph_data:
        fold_convert_our_data_graph(data, data_x)
        CAIL_save_dataset = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/raw_CAIL_dataset.json"
        with open(CAIL_save_dataset, 'w') as f:
            for item in data:
                f.writelines(json.dumps(item, ensure_ascii=False))
                f.write('\n')

    elif generate_rationale:
        CAIL_save_dataset_rationale = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/rationale_CAIL_dataset.json"
        results = generate_rationale_func(data, data_x)
        with open(CAIL_save_dataset_rationale, 'w') as f:
            for item in results:
                f.writelines(json.dumps(item, ensure_ascii=False))
                f.write('\n')
    else:
        da_type = "graph"
        match_data_seq2seq_json = '/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/match_data_seq2seq_{}_wolaw.json'.format(
            da_type)
        midmatch_data_seq2seq_json = '/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/midmatch_data_seq2seq_{}_wolaw.json'.format(
            da_type)
        dismatch_data_seq2seq_json = '/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/CAIL/dismatch_data_seq2seq_{}_wolaw.json'.format(
            da_type)
        convert(match_data_seq2seq_json, data, data_x, type='match', generate_mode=da_type)
        convert(midmatch_data_seq2seq_json, data, data_x, type='midmatch', generate_mode=da_type)
        convert(dismatch_data_seq2seq_json, data, data_x, type='dismatch', generate_mode=da_type)

        print(u'输出over！')
