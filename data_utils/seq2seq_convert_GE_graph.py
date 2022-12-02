import sys

from pip._vendor.distlib.database import make_graph

sys.path.append("..")
from models.selector_two_multi_class_ot_v3 import Selector2_mul_class, args, load_checkpoint, OT, load_data, data_extract_npy, data_extract_json, device
import torch
from utils.snippets import *
import torch
import json
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
    AO_a, YO_a, ZO_a, AI_a, YI_a, ZI_a = [], [], [], [], [], []
    AO_b, YO_b, ZO_b, AI_b, YI_b, ZI_b = [], [], [], [], [], []

    output_batch_A, batch_mask_A = model(case_A)
    output_batch_B, batch_mask_B = model(case_B)
    plan_list = OT_model(output_batch_A, output_batch_B, case_A, case_B, None,
                            batch_mask_A, batch_mask_B, 'valid')
    OT_matrix = torch.ge(plan_list, 1 / case_A.shape[1] / args.threshold_ot).long()


    """
    根据这个取出来谁跟谁对齐了
    出现三个类型的边，<pair> <single> law 这个通过graph那块处理出来吧，这边记得对应上id
    """


    vec_correct_A = torch.argmax(output_batch_A, dim=-1).long()[0][:seq_len_A]
    vec_correct_B = torch.argmax(output_batch_B, dim=-1).long()[0][:seq_len_B]
    relation_A = torch.sum(OT_matrix[0], dim=1)
    relation_B = torch.sum(OT_matrix[0], dim=0)
    pair_sentences = []

    for i, label in enumerate(vec_correct_A):
        if label == 1:
            if relation_A[i] >= 1:
                AI_a.append(i)
            else:
                AO_a.append(i)
        elif label == 2:
            if relation_A[i] >= 1:
                YI_a.append(i)
            else:
                YO_a.append(i)
        elif label == 3:
            if relation_A[i] >= 1:
                ZI_a.append(i)
            else:
                ZO_a.append(i)

    for i, label in enumerate(vec_correct_B):
        if label == 1:
            if relation_B[i] >= 1:
                AI_b.append(i)
            else:
                AO_b.append(i)
        elif label == 2:
            if relation_B[i] >= 1:
                YI_b.append(i)
            else:
                YO_b.append(i)
        elif label == 3:
            if relation_B[i] >= 1:
                ZI_b.append(i)
            else:
                ZO_b.append(i)
    for a in range(seq_len_A):
        for b in range(seq_len_B):
            if OT_matrix[0][a][b]:
                if vec_correct_A[a] and vec_correct_B[b]:
                    pair_sentences.append([a, b])
    AO, YO, ZO, AI, YI, ZI = [AO_a, AO_b], [YO_a, YO_b], [ZO_a, ZO_b], [AI_a, AI_b], [YI_a, YI_b], [ZI_a, ZI_b]

    return AO, YO, ZO, AI, YI, ZI, [vec_correct_A+(torch.ge(relation_A[:seq_len_A], 1)*3)*vec_correct_A, vec_correct_B+(torch.ge(relation_B[:seq_len_B], 1)*3)*vec_correct_B], pair_sentences




def generate_text_cluster(case_a, case_b, d, AO, YO, ZO, AI, YI, ZI, A_all_true, Y_all_true, Z_all_true, AI_true, YI_true, ZI_true):
    source_1_a = ''.join(["[AO]" + case_a[0][i] for i in AO[0]] + ["[YO]" + case_a[0][i] for i in YO[0]] +
                         ["[ZO]" + case_a[0][i] for i in ZO[0]] + ["[AI]" + case_a[0][i] for i in AI[0]] +
                         ["[YI]" + case_a[0][i] for i in YI[0]] + ["[ZI]" + case_a[0][i] for i in ZI[0]])

    source_1_b = ''.join(["[AO]" + case_b[0][i] for i in AO[1]] + ["[YO]" + case_b[0][i] for i in YO[1]] +
                         ["[ZO]" + case_b[0][i] for i in ZO[1]] + ["[AI]" + case_b[0][i] for i in AI[1]] +
                         ["[YI]" + case_b[0][i] for i in YI[1]] + ["[ZI]" + case_b[0][i] for i in ZI[1]])

    source_2_a = ''.join(
        ["[AO]" + case_a[0][i] for i in A_all_true[0] if i not in AI_true[0]] + ["[YO]" + case_a[0][i] for i in
                                                                                 Y_all_true[0] if i not in YI_true[0]] +
        ["[ZO]" + case_a[0][i] for i in Z_all_true[0] if i not in ZI_true[0]] + ["[AI]" + case_a[0][i] for i in
                                                                                 AI_true[0]] +
        ["[YI]" + case_a[0][i] for i in YI_true[0]] + ["[ZI]" + case_a[0][i] for i in ZI_true[0]])

    source_2_b = ''.join(
        ["[AO]" + case_b[0][i] for i in A_all_true[1] if i not in AI_true[1]] + ["[YO]" + case_b[0][i] for i in
                                                                                 Y_all_true[1] if i not in YI_true[1]] +
        ["[ZO]" + case_b[0][i] for i in Z_all_true[1] if i not in ZI_true[1]] + ["[AI]" + case_b[0][i] for i in
                                                                                 AI_true[1]] +
        ["[YI]" + case_b[0][i] for i in YI_true[1]] + ["[ZI]" + case_b[0][i] for i in ZI_true[1]])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': '；'.join(list(d['explanation'].values())),
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result
import re
xingfa_dic = "/home/zhongxiang_sun/code/explanation_project/explanation_model/models_v2/data/xingfa_law_dic.json"

def get_law_text(text):
    tiao_text_A = ""
    a = r'《(.*?)》'
    with open(xingfa_dic, 'r') as f:
        xingfalaws = json.load(f)
    law = re.findall(a, text)[0]
    if law == "中华人民共和国刑法":
        tiao = text.split("》")[1]
        if tiao in xingfalaws.keys():
            tiao_text_A = xingfalaws[tiao]
    return tiao_text_A

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
        if law == "中华人民共和国刑法":
            xingfa_laws.append(law_content)
    return xingfa_laws

def generate_text_GE_graph(case_a, case_b, d, AO, YO, ZO, AI, YI, ZI, A_all_true, Y_all_true, Z_all_true, AI_true, YI_true, ZI_true, prediction_relation, label_realtion):


    def generate_source_a_b(relation, AI, YI, ZI):
        inter_pairs = []
        source_1_a_b = ""
        for item in relation:
            a, b = item
            if a in AI[0]:
                source_1_a_b += '[AI]'+case_a[0][a]
            elif a in YI[0]:
                source_1_a_b += '[YI]'+case_a[0][a]
            elif a in ZI[0]:
                source_1_a_b += '[ZI]'+case_a[0][a]
            else:
                print("error at generate_text_GE_graph")
            source_1_a_b += '[R]'

            if b in AI[1]:
                source_1_a_b += '[AI]'+case_b[0][b]
            elif b in YI[1]:
                source_1_a_b += '[YI]'+case_b[0][b]
            elif b in ZI[1]:
                source_1_a_b += '[ZI]'+case_b[0][b]
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
    source_1_a_b, inter_pairs = generate_source_a_b(prediction_relation, AI, YI, ZI)
    source_1_a = ''.join(["[AO]" + case_a[0][i] for i in AO[0]] + ["[YO]" + case_a[0][i] for i in YO[0]] +
                         ["[ZO]" + case_a[0][i] for i in ZO[0]])

    source_1_b = ''.join(["[AO]" + case_b[0][i] for i in AO[1]] + ["[YO]" + case_b[0][i] for i in YO[1]] +
                         ["[ZO]" + case_b[0][i] for i in ZO[1]])

    source_2_a_b, _ = generate_source_a_b(label_realtion, AI_true, YI_true, ZI_true)
    source_2_a = ''.join(
        ["[AO]" + case_a[0][i] for i in A_all_true[0] if i not in AI_true[0]] + ["[YO]" + case_a[0][i] for i in
                                                                                 Y_all_true[0] if i not in YI_true[0]] +
        ["[ZO]" + case_a[0][i] for i in Z_all_true[0] if i not in ZI_true[0]] )

    source_2_b = ''.join(
        ["[AO]" + case_b[0][i] for i in A_all_true[1] if i not in AI_true[1]] + ["[YO]" + case_b[0][i] for i in
                                                                                 Y_all_true[1] if i not in YI_true[1]] +
        ["[ZO]" + case_b[0][i] for i in Z_all_true[1] if i not in ZI_true[1]])

    result = {
        'source_1': source_1_a_b + source_1_a + source_1_b,
        'source_2': source_2_a_b + source_2_a + source_2_b,
        'explanation': '；'.join(list(d['explanation'].values())),
        'source_1_dis': [source_1_a, source_1_b, source_1_a_b, source_a_b_law],
        'source_2_dis': [source_2_a, source_2_b, source_2_a_b, source_a_b_law],
        'label': d['label'],
        "id":d['id']
    }
    return result, inter_pairs


def get_extract_text(case_a, prediction):
    source_1_a = ''
    for i, output_class in enumerate(prediction):
        if output_class == 1:
            source_1_a += "[AO]" + case_a[0][i]
        elif output_class == 2:
            source_1_a += "[YO]" + case_a[0][i]
        elif output_class == 3:
            source_1_a += "[ZO]" + case_a[0][i]
        elif output_class == 4:
            source_1_a += "[AI]" + case_a[0][i]
        elif output_class == 5:
            source_1_a += "[YI]" + case_a[0][i]
        elif output_class == 6:
            source_1_a += "[ZI]" + case_a[0][i]
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
        'explanation': '；'.join(list(d['explanation'].values())),
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result


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

def generate_text_wo_token(case_a, case_b, d, prediction, label):

    source_1_a = get_extract_text_wo_token(case_a, prediction[0])
    source_1_b = get_extract_text_wo_token(case_b, prediction[1])
    source_2_a = get_extract_text_wo_token(case_a, label[0])
    source_2_b = get_extract_text_wo_token(case_b, label[1])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': '；'.join(list(d['explanation'].values())),
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result


def fold_convert_our_data_ot(data, data_x, type, generate=False, generate_mode = 'cluster'):
    """每一fold用对应的模型做数据转换
    """

    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-10-simot-1-simpercent-1.pkl")
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract_ot-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-10-simot-1-simpercent-1.pkl")
        results = []
        print(type+"ing")
        for i, d in enumerate(data):
            if type == 'match' and d["label"] == 2 or type == 'midmatch' and d["label"] == 1 or type == 'dismatch' and d["label"] == 0:
                case_a = d['case_A']
                case_b = d['case_B']
                important_A, important_B = [], []
                data_y_seven_class_A, data_y_seven_class_B = [0]*len(case_a[0]), [0]*len(case_b[0])
                for pos_list in d['relation_label'].values():
                    for pos in pos_list:
                        row, col = pos[0], pos[-1]
                        important_A.append(row)
                        important_B.append(col)

                for j in case_a[1]:
                    if j[0] in important_A:
                        data_y_seven_class_A[j[0]] = j[1] + 3
                    else:
                        data_y_seven_class_A[j[0]] = j[1]

                for j in case_b[1]:
                    if j[0] in important_B:
                        data_y_seven_class_B[j[0]] = j[1] + 3
                    else:
                        data_y_seven_class_B[j[0]] = j[1]
                label = [data_y_seven_class_A, data_y_seven_class_B]



                AO, YO, ZO, AI, YI, ZI, prediction, pair_sentence = model_class(model, ot_model, torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device),
                                                     torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device),len(case_a[0]), len(case_b[0]))

                A_all_true, Y_all_true, Z_all_true, AI_true, YI_true, ZI_true = [], [], [], [], [], []
                label_pair_sentence = d['relation_label']['relation_label_aqss'] +  d['relation_label']['relation_label_yjss'] \
                                          + d['relation_label']['relation_label_zyjd']
                temp_a, temp_b = [], []
                for i in d['relation_label']['relation_label_aqss']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                AI_true.append(temp_a)
                AI_true.append(temp_b)

                temp_a, temp_b = [], []
                for i in d['relation_label']['relation_label_yjss']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                YI_true.append(temp_a)
                YI_true.append(temp_b)

                temp_a, temp_b = [], []
                for i in d['relation_label']['relation_label_zyjd']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                ZI_true.append(temp_a)
                ZI_true.append(temp_b)
                aqss_temp, yjss_temp, zyjd_temp = [], [], []
                for i in case_a[1]:
                    if i[1] == 1:
                        aqss_temp.append(i[0])
                    elif i[1] == 2:
                        yjss_temp.append(i[0])
                    elif i[1] == 3:
                        zyjd_temp.append(i[0])
                A_all_true.append(aqss_temp)
                Y_all_true.append(yjss_temp)
                Z_all_true.append(zyjd_temp)

                aqss_temp, yjss_temp, zyjd_temp = [], [], []
                for i in case_b[1]:
                    if i[1] == 1:
                        aqss_temp.append(i[0])
                    elif i[1] == 2:
                        yjss_temp.append(i[0])
                    elif i[1] == 3:
                        zyjd_temp.append(i[0])
                A_all_true.append(aqss_temp)
                Y_all_true.append(yjss_temp)
                Z_all_true.append(zyjd_temp)
                if generate:
                    if generate_mode == 'cluster':
                        results.append(generate_text_cluster(case_a, case_b, d, AO, YO, ZO, AI, YI, ZI, A_all_true, Y_all_true, Z_all_true, AI_true, YI_true,
                                      ZI_true))
                    elif generate_mode == 'sort':
                        results.append(generate_text_sort(case_a, case_b, d, prediction, label))
                    elif generate_mode == 'graph':
                        result, inter_pairs = generate_text_GE_graph(case_a, case_b, d, AO, YO, ZO, AI, YI, ZI, A_all_true, Y_all_true,
                                               Z_all_true, AI_true, YI_true,
                                               ZI_true, pair_sentence, label_pair_sentence)

                        results.append(result)
                    else:
                        results.append(generate_text_wo_token(case_a, case_b, d, prediction, label))

        if generate:
            return results

def generate_rationale_func(data, data_x):
    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-10-simot-1-simpercent-1.pkl")
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract_ot-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-10-simot-1-simpercent-1.pkl")
        results = []
        for i, d in enumerate(data):
            case_a = d['case_A']
            case_b = d['case_B']
            important_A, important_B = [], []
            data_y_seven_class_A, data_y_seven_class_B = [0]*len(case_a[0]), [0]*len(case_b[0])
            for pos_list in d['relation_label'].values():
                for pos in pos_list:
                    row, col = pos[0], pos[-1]
                    important_A.append(row)
                    important_B.append(col)

            for j in case_a[1]:
                if j[0] in important_A:
                    data_y_seven_class_A[j[0]] = j[1] + 3
                else:
                    data_y_seven_class_A[j[0]] = j[1]

            for j in case_b[1]:
                if j[0] in important_B:
                    data_y_seven_class_B[j[0]] = j[1] + 3
                else:
                    data_y_seven_class_B[j[0]] = j[1]
            label = [data_y_seven_class_A, data_y_seven_class_B]



            AO, YO, ZO, AI, YI, ZI, prediction, pair_sentence = model_class(model, ot_model, torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device),
                                                 torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device),len(case_a[0]), len(case_b[0]))


            results.append(generate_rationale_text(case_a, case_b, d, prediction, label))

        return results


def fold_convert_our_data_graph(data, data_x):
    """每一fold用对应的模型做数据转换
    """

    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-10-simot-1-simpercent-1.pkl")
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract_ot-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-10-simot-1-simpercent-1.pkl")
        for i, d in enumerate(data):
                case_a = d['case_A']
                case_b = d['case_B']
                important_A, important_B = [], []
                data_y_seven_class_A, data_y_seven_class_B = [0]*len(case_a[0]), [0]*len(case_b[0])
                for pos_list in d['relation_label'].values():
                    for pos in pos_list:
                        row, col = pos[0], pos[-1]
                        important_A.append(row)
                        important_B.append(col)

                for j in case_a[1]:
                    if j[0] in important_A:
                        data_y_seven_class_A[j[0]] = j[1] + 3
                    else:
                        data_y_seven_class_A[j[0]] = j[1]

                for j in case_b[1]:
                    if j[0] in important_B:
                        data_y_seven_class_B[j[0]] = j[1] + 3
                    else:
                        data_y_seven_class_B[j[0]] = j[1]
                label = [data_y_seven_class_A, data_y_seven_class_B]



                AO, YO, ZO, AI, YI, ZI, prediction, pair_sentence = model_class(model, ot_model, torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device),
                                                     torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device),len(case_a[0]), len(case_b[0]))

                A_all_true, Y_all_true, Z_all_true, AI_true, YI_true, ZI_true = [], [], [], [], [], []
                label_pair_sentence = d['relation_label']['relation_label_aqss'] + d['relation_label']['relation_label_yjss'] \
                                          + d['relation_label']['relation_label_zyjd']
                temp_a, temp_b = [], []
                for i in d['relation_label']['relation_label_aqss']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                AI_true.append(temp_a)
                AI_true.append(temp_b)

                temp_a, temp_b = [], []
                for i in d['relation_label']['relation_label_yjss']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                YI_true.append(temp_a)
                YI_true.append(temp_b)

                temp_a, temp_b = [], []
                for i in d['relation_label']['relation_label_zyjd']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                ZI_true.append(temp_a)
                ZI_true.append(temp_b)
                aqss_temp, yjss_temp, zyjd_temp = [], [], []
                for i in case_a[1]:
                    if i[1] == 1:
                        aqss_temp.append(i[0])
                    elif i[1] == 2:
                        yjss_temp.append(i[0])
                    elif i[1] == 3:
                        zyjd_temp.append(i[0])
                A_all_true.append(aqss_temp)
                Y_all_true.append(yjss_temp)
                Z_all_true.append(zyjd_temp)

                aqss_temp, yjss_temp, zyjd_temp = [], [], []
                for i in case_b[1]:
                    if i[1] == 1:
                        aqss_temp.append(i[0])
                    elif i[1] == 2:
                        yjss_temp.append(i[0])
                    elif i[1] == 3:
                        zyjd_temp.append(i[0])
                A_all_true.append(aqss_temp)
                Y_all_true.append(yjss_temp)
                Z_all_true.append(zyjd_temp)

                _, inter_pairs = generate_text_GE_graph(case_a, case_b, d, AO, YO, ZO, AI, YI, ZI, A_all_true, Y_all_true,
                                       Z_all_true, AI_true, YI_true,
                                       ZI_true, pair_sentence, label_pair_sentence)

                d["inter_pairs"] = inter_pairs
                d["rationale_a"] = AO[0]+YO[0]+ZO[0]+AI[0]+YI[0]+ZI[0]
                d["rationale_b"] = AO[1]+YO[1]+ZO[1]+AI[1]+YI[1]+ZI[1]
                for item in d["inter_pairs"]:
                    if item[0] not in d["rationale_a"] or item[1] not in d["rationale_b"]:
                            print(1)





def convert(filename, data, data_x, type,  generate_mode):
    """转换为生成式数据
    """
    total_results = fold_convert_our_data_ot(data, data_x, type, generate=True, generate_mode=generate_mode)

    with open(filename, 'w') as f:
        for item in total_results:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')



if __name__ == '__main__':

    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    make_graph_data = False
    generate_rationale = True
    if make_graph_data:
        fold_convert_our_data_graph(data, data_x)
        ELAM_save_dataset = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/raw_ELAM_dataset.json"
        with open(ELAM_save_dataset, 'w') as f:
            for item in data:
                f.writelines(json.dumps(item, ensure_ascii=False))
                f.write('\n')
    elif generate_rationale:
        ELAM_save_dataset_rationale = "/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/rationale_ELAM_dataset.json"
        results = generate_rationale_func(data, data_x)
        with open(ELAM_save_dataset_rationale, 'w') as f:
            for item in results:
                f.writelines(json.dumps(item, ensure_ascii=False))
                f.write('\n')


    else:
        da_type = "graph"
        match_data_seq2seq_json = '/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/ELAM/match_data_seq2seq_{}_wolaw.json'.format(da_type)
        midmatch_data_seq2seq_json = '/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/ELAM/midmatch_data_seq2seq_{}_wolaw.json'.format(da_type)
        dismatch_data_seq2seq_json = '/home/zhongxiang_sun/code/explanation_project/GELegal_Match/data/ELAM/dismatch_data_seq2seq_{}_wolaw.json'.format(da_type)
        convert(match_data_seq2seq_json, data, data_x, type='match', generate_mode=da_type)
        convert(midmatch_data_seq2seq_json, data, data_x, type='midmatch',  generate_mode=da_type)
        convert(dismatch_data_seq2seq_json, data, data_x, type='dismatch',  generate_mode=da_type)


        print(u'输出over！')
