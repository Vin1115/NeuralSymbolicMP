from collections import defaultdict
import pickle

t_t_dict = {
    "r1(s1,f1)": "1p",
    "(r1(s1,e1))&(r2(e1,f1))": "2p",
    "(r1(s1,e1))&((r2(e1,e2))&(r3(e2,f1)))": "3p",
    "(r1(s1,f1))&(r2(s2,f1))": "2i",
    "(r1(s1,f1))&((r2(s2,f1))&(r3(s3,f1)))": "3i",
    "(r1(s1,e1))&((r2(s2,e1))&(r3(e1,f1)))": "ip",
    "(r1(s1,e1))&((r2(e1,f1))&(r3(s2,f1)))": "pi",
    "(r1(s1,f1))&(!(r2(s2,f1)))": "2in",
    "((r1(s1,f1))&(r2(s2,f1)))&(!(r3(s3,f1)))": "3in",
    "((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f1))": "inp",
    "((r1(s1,e1))&(r2(e1,f1)))&(!(r3(s2,f1)))": "pin",
    "(r1(s1,f1))|(r2(s2,f1))": "2u",
    "((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))": "up",
    "((r1(s1,e1))&(!(r2(e1,f1))))&(r3(s2,f1))": "pni",
    "(r1(s1,e1))&((r2(e1,f1))&(r3(e1,f1)))": "2m",
    "(r1(s1,e1))&((r2(e1,f1))&(!(r3(e1,f1))))": "2nm",
    "(r1(s1,e1))&((r2(s2,e1))&((r3(e1,f1))&(r4(e1,f1))))": "im",
    "(r1(s1,e1))&((r2(e1,e2))&((r3(e1,e2))&(r4(e2,f1))))": "3mp",
    "(r1(s1,e1))&((r2(e1,e2))&((r3(e2,f1))&(r4(e2,f1))))": "3pm",
    "(r1(s1,f1))&(r2(e1,f1))": "2il",
    "(r1(s1,f1))&((r2(s2,f1))&(r3(e1,f1)))": "3il",
    "(r1(s1,e1))&((r2(s2,e2))&((r3(e1,e2))&((r4(e1,f1))&(r5(e2,f1)))))": "3c",
    "(r1(s1,e1))&((r2(s2,e2))&((r3(e1,f1))&((r4(e1,f1))&((r5(e1,e2))&(r6(e2,f1))))))": "3cm"
}

group1_keys = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
group2_keys = ["pni", "2m", "2nm", "im", "3mp", "3pm", "2il", "3il", "3c", "3cm"]
group3_keys = ["2in", "3in", "inp", "pin"]

group1_mrr = defaultdict(float)
group1_count = defaultdict(int)
group2_mrr = defaultdict(float)
group2_count = defaultdict(int)
group3_mrr = defaultdict(float)
group3_count = defaultdict(int)

with open('EFO-1_log/FB15k-237-EFO1_result/nsmp_lambda0.3_alpha100_ds1/all_logging.pickle', 'rb') as file:
    data = pickle.load(file)
    print(data)

for key, value in data.items():
    for sub_key, metrics_list in value.items():
        if sub_key in t_t_dict:
            metrics = metrics_list[0]
            num_queries = metrics['num_queries']
            mrr = metrics['MRR'] / num_queries
            hits1 = metrics['HITS1'] / num_queries
            hits3 = metrics['HITS3'] / num_queries
            hits10 = metrics['HITS10'] / num_queries
            template_type = t_t_dict[sub_key]

            print(f"{template_type} MRR: {mrr:.4f} HITS1: {hits1:.4f} HITS3: {hits3:.4f} HITS10: {hits10:.4f} NUMS {num_queries:.1f}")

            if template_type in group1_keys:
                group1_mrr[template_type] += mrr
                group1_count[template_type] += 1
            elif template_type in group2_keys:
                group2_mrr[template_type] += mrr
                group2_count[template_type] += 1
            elif template_type in group3_keys:
                group3_mrr[template_type] += mrr
                group3_count[template_type] += 1

if sum(group1_count.values()) > 0:
    avg_mrr_group1 = sum(group1_mrr.values()) / sum(group1_count.values())
    print(f"Average MRR for Group 1: {avg_mrr_group1:.4f}")
else:
    print("No data for Group 1")

if sum(group2_count.values()) > 0:
    avg_mrr_group2 = sum(group2_mrr.values()) / sum(group2_count.values())
    print(f"Average MRR for Group 2: {avg_mrr_group2:.4f}")
else:
    print("No data for Group 2")

if sum(group3_count.values()) > 0:
    avg_mrr_group3 = sum(group3_mrr.values()) / sum(group3_count.values())
    print(f"Average MRR for Group 3: {avg_mrr_group3:.4f}")
else:
    print("No data for Group 3")
