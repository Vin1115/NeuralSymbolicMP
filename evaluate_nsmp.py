import argparse
import json
import os
import os.path as osp
import pickle
from collections import defaultdict

import pandas as pd
import torch
import tqdm

from src.pipeline import (Reasoner, NeuralSymbolicMPReasoner, NeuralSymbolicMPLayer, 
                          NeuralSymbolicMPWithoutDynamicPruningReasoner, NeuralSymbolicMPWithoutDynamicPruningLayer)
from src.structure import get_nbp_class
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import QueryAnsweringSeqDataLoader
from src.utils.util import set_global_seed
from src.utils.data import AdjMatData

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cuda:0")

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--eval_queries", default="eval_queries.csv",  action='append')
parser.add_argument("--batch_size_eval_dataloader", type=int, default=1, help="batch size for evaluation")

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--checkpoint_path", default="pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt", type=str, help="path to the KGE checkpoint")
parser.add_argument("--embedding_dim", type=int, default=1000)
parser.add_argument("--margin", type=float, default=10)
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--p", type=int, default=1)

# mp-based reasoning machine
parser.add_argument("--reasoner", type=str, default='nsmp', choices=['nsmp', 'nsmpwodp'])
parser.add_argument("--depth_shift", type=int, default=1)
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--llambda", default=0.3, type=float, help="")
parser.add_argument("--prefix", type=str, default='nsmp_lambda0.3_alpha100_ds1')
parser.add_argument("--alpha", default=100, type=int, help="")


def ranking2metrics(ranking, easy_ans, hard_ans, ranking_device):
    num_hard = len(hard_ans)
    num_easy = len(easy_ans)
    assert len(set(hard_ans).intersection(set(easy_ans))) == 0
    # only take those answers' rank
    cur_ranking = ranking[list(easy_ans) + list(hard_ans)]
    cur_ranking, indices = torch.sort(cur_ranking)
    masks = indices >= num_easy
    answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(ranking_device)
    cur_ranking = cur_ranking - answer_list + 1
    # filtered setting: +1 for start at 0, -answer_list for ignore other answers
    cur_ranking = cur_ranking[masks]
    # only take indices that belong to the hard answers
    mrr = torch.mean(1. / cur_ranking).item()
    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
    h10 = torch.mean(
        (cur_ranking <= 10).to(torch.float)).item()
    return mrr, h1, h3, h10


# def neural_symbolic_evaluate_dnf(
#         e,
#         desc,
#         dataloader,
#         nbp: NeuralBinaryPredicate,
#         reasoner: Reasoner):
#     '''
#     A bad implementation. It should be used in the code released by LMPNN or CLMPT.
#     '''
#     metric = defaultdict(lambda: defaultdict(list))
#     fofs = dataloader.get_fof_list()
#
#     # conduct reasoning
#     with tqdm.tqdm(fofs, desc=desc) as t:
#         for fof in t:
#             with torch.no_grad():
#                 if fof.lstr == '(r1(s1,f))|(r2(s2,f))':
#                     # query1
#                     lformula1 = parse_lstr_to_lformula('r1(s1,f)')
#                     query1 = EFO1Query(lformula1)
#                     query1.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s1']
#                     query1.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
#                     query1.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r1']
#                     query1.easy_answer_list = fof.easy_answer_list
#                     query1.hard_answer_list = fof.hard_answer_list
#                     query1.noisy_answer_list = fof.noisy_answer_list
#                     reasoner.initialize_with_query(query1)
#                     reasoner.estimate_variable_embeddings()
#                     batch_fvar_emb1 = reasoner.get_ent_emb('f')
#                     batch_fvar_vec1 = reasoner.get_free_vec('f')
#                     # query2
#                     lformula2 = parse_lstr_to_lformula('r1(s1,f)')
#                     query2 = EFO1Query(lformula2)
#                     query2.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s2']
#                     query2.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
#                     query2.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r2']
#                     query2.easy_answer_list = fof.easy_answer_list
#                     query2.hard_answer_list = fof.hard_answer_list
#                     query2.noisy_answer_list = fof.noisy_answer_list
#                     reasoner.initialize_with_query(query2)
#                     reasoner.estimate_variable_embeddings()
#                     batch_fvar_emb2 = reasoner.get_ent_emb('f')
#                     batch_fvar_vec2 = reasoner.get_free_vec('f')
#                     batch_entity_rankings = nbp.get_all_entity_rankings_ns_dnf(batch_fvar_emb1, batch_fvar_emb2,
#                                                                                batch_fvar_vec1,
#                                                                                batch_fvar_vec2, score="cos",
#                                                                                neural_value=1 - args.llambda,
#                                                                                symbolic_value=args.llambda)
#                 elif fof.lstr == '((r1(s1,e1))|(r2(s2,e1)))&(r3(e1,f))' or fof.lstr == '((r1(s1,e1))&(r3(e1,f)))|((r2(s2,e1))&(r3(e1,f)))':
#                     # query1
#                     lformula1 = parse_lstr_to_lformula('(r1(s1,e1))&(r2(e1,f))')  # (r1(s1,e1))&(r3(e1,f))
#                     query1 = EFO1Query(lformula1)
#                     query1.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s1']
#                     query1.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
#                     query1.term_grounded_entity_id_dict['e1'] = fof.term_grounded_entity_id_dict['e1']
#                     query1.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r1']
#                     query1.pred_grounded_relation_id_dict['r2'] = fof.pred_grounded_relation_id_dict['r3']
#                     query1.easy_answer_list = fof.easy_answer_list
#                     query1.hard_answer_list = fof.hard_answer_list
#                     query1.noisy_answer_list = fof.noisy_answer_list
#                     reasoner.initialize_with_query(query1)
#                     reasoner.estimate_variable_embeddings()
#                     batch_fvar_emb1 = reasoner.get_ent_emb('f')
#                     batch_fvar_vec1 = reasoner.get_free_vec('f')
#                     # query2
#                     lformula2 = parse_lstr_to_lformula('(r1(s1,e1))&(r2(e1,f))')  # (r2(s2,e1))&(r3(e1,f))
#                     query2 = EFO1Query(lformula2)
#                     query2.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s2']
#                     query2.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
#                     query2.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r2']
#                     query2.term_grounded_entity_id_dict['e1'] = fof.term_grounded_entity_id_dict['e1']
#                     query2.pred_grounded_relation_id_dict['r2'] = fof.pred_grounded_relation_id_dict['r3']
#                     query2.easy_answer_list = fof.easy_answer_list
#                     query2.hard_answer_list = fof.hard_answer_list
#                     query2.noisy_answer_list = fof.noisy_answer_list
#                     reasoner.initialize_with_query(query2)
#                     reasoner.estimate_variable_embeddings()
#                     batch_fvar_emb2 = reasoner.get_ent_emb('f')
#                     batch_fvar_vec2 = reasoner.get_free_vec('f')
#                     batch_entity_rankings = nbp.get_all_entity_rankings_dnf11(batch_fvar_emb1, batch_fvar_emb2,
#                                                                               batch_fvar_vec1,
#                                                                               batch_fvar_vec2, score="cos",
#                                                                               neural_value=1 - args.llambda,
#                                                                               symbolic_value=args.llambda)
#                 else:
#                     reasoner.initialize_with_query(fof)
#                     reasoner.estimate_variable_embeddings()
#                     batch_fvar_emb = reasoner.get_ent_emb('f')
#                     batch_fvar_vec = reasoner.get_free_vec('f')
#
#                     batch_entity_rankings = nbp.get_all_entity_rankings_ns(
#                         batch_fvar_emb, batch_fvar_vec, score="cos",
#                         neural_value=1 - args.llambda, symbolic_value=args.llambda)
#
#             # [batch_size, num_entities]
#             compute_evaluation_scores(
#                 fof, batch_entity_rankings, metric[fof.lstr])
#             t.set_postfix({'lstr': fof.lstr})
#
#             sum_metric = defaultdict(dict)
#             for lstr in metric:
#                 for score_name in metric[lstr]:
#                     sum_metric[lstr2name[lstr]][score_name] = float(
#                         np.mean(metric[lstr][score_name]))
#
#         postfix = {}
#         for name in ['1p', '2p', '3p', '2i', 'inp']:
#             if name in sum_metric:
#                 postfix[name + '_hit3'] = sum_metric[name]['hit3']
#         torch.cuda.empty_cache()
#
#     sum_metric['epoch'] = e
#     logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")


def neural_symbolic_evaluate(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        args):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    two_marginal_logs = defaultdict(float)
    one_marginal_logs, no_marginal_logs = defaultdict(float), defaultdict(float)
    fofs = dataloader.get_fof_list()
    formula = list(dataloader.lstr_iterator.keys())[0]
    num_var = (1 if "f1" in formula else 0) + (1 if "f2" in formula else 0)
    f_str_list = [f'f{i + 1}' for i in range(num_var)]
    f_str = '_'.join(f_str_list)
    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            with torch.no_grad():
                ranking_list = []
                reasoner.initialize_with_query(fof, formula)  # TODO
                reasoner.estimate_variable_embeddings()
                batch_fvar_emb = reasoner.get_ent_emb('f1')
                batch_fvar_vec = reasoner.get_free_vec('f1')
                batch_entity_rankings = nbp.get_all_entity_rankings_ns(
                    batch_fvar_emb, batch_fvar_vec, score="cos", neural_value=1-args.llambda, symbolic_value=args.llambda)
                ranking_list.append(batch_entity_rankings)

                ranking = torch.cat(ranking_list, dim=1)
                # create a new torch Tensor for batch_entity_range
                # achieve the ranking of all entities
                for i in range(ranking.shape[0]):
                    easy_ans = [instance[0] for instance in fof.easy_answer_list[i][f_str]]
                    hard_ans = [instance[0] for instance in fof.hard_answer_list[i][f_str]]
                    mrr, h1, h3, h10 = ranking2metrics(ranking[i], easy_ans, hard_ans, nbp.device)
                    two_marginal_logs['MRR'] += mrr
                    two_marginal_logs['HITS1'] += h1
                    two_marginal_logs['HITS3'] += h3
                    two_marginal_logs['HITS10'] += h10
                    two_marginal_logs["num_queries"] += 1

    return two_marginal_logs, one_marginal_logs, no_marginal_logs


if __name__ == "__main__":
    args = parser.parse_args()
    set_global_seed(args.seed)

    kgidx = KGIndex.load(
        osp.join(args.task_folder, "kgindex.json"))

    print(f"loading the nbp {args.model_name}")
    nbp = get_nbp_class(args.model_name)(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        scale=args.scale,
        device=args.device)

    base_data = AdjMatData(args.task_folder)
    mat = base_data.rel_mat

    if args.checkpoint_path:
        print("loading model from", args.checkpoint_path)
        nbp.load_state_dict(torch.load(args.checkpoint_path), strict=True)

    nbp.to(args.device)
    print(f"model loaded from {args.checkpoint_path}")

    print("loading dataset")
    info_eval_queries = pd.read_csv(osp.join('data', 'eval_queries.csv'))  #
    eval_queries = list(info_eval_queries.formula)
    eval_formula_id = list(info_eval_queries.formula_id)
    print("eval queries", eval_queries)

    if args.reasoner == 'nsmpwodp': 
        lgnn_layer = NeuralSymbolicMPWithoutDynamicPruningLayer(nbp=nbp, mat=mat, alpha=args.alpha)
        lgnn_layer.to(nbp.device)

        reasoner = NeuralSymbolicMPWithoutDynamicPruningReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)

        all_log = defaultdict(dict)
        for i in range(len(eval_queries)):
            test_dataloader = QueryAnsweringSeqDataLoader(
                osp.join(args.task_folder, f'test_{eval_formula_id[i]}_real_EFO1_qaa.json'),
                target_lstr=eval_queries,
                batch_size=args.batch_size_eval_dataloader,
                shuffle=False,
                num_workers=0)
            two_marginal_logs, one_marginal_logs, no_marginal_logs = neural_symbolic_evaluate(0, f"NN evaluate test set ",
                                        test_dataloader, nbp, reasoner, args)
            log = {eval_queries[i] : [two_marginal_logs, one_marginal_logs, no_marginal_logs]}
            all_log[eval_queries[i]] = log
            if osp.exists('EFO-1_log/{}_result/{}'.format(args.task_folder.split("/")[-1], args.prefix)) == False:
                os.makedirs('EFO-1_log/{}_result/{}'.format(args.task_folder.split("/")[-1], args.prefix))
            with open('EFO-1_log/{}_result/{}/all_logging_test_0_{}.pickle'.format(args.task_folder.split("/")[-1], args.prefix, eval_formula_id[i]), 'wb') as f:
                    pickle.dump(log, f)
        with open('EFO-1_log/{}_result/{}/all_logging.pickle'.format(args.task_folder.split("/")[-1], args.prefix), 'wb') as g:
                pickle.dump(all_log, g)
    elif args.reasoner == 'nsmp': 
        lgnn_layer = NeuralSymbolicMPLayer(nbp=nbp, mat=mat, alpha=args.alpha)
        lgnn_layer.to(nbp.device)

        reasoner = NeuralSymbolicMPReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)

        all_log = defaultdict(dict)
        for i in range(len(eval_queries)):
            test_dataloader = QueryAnsweringSeqDataLoader(
                osp.join(args.task_folder, f'test_{eval_formula_id[i]}_real_EFO1_qaa.json'),
                target_lstr=eval_queries,
                batch_size=args.batch_size_eval_dataloader,
                shuffle=False,
                num_workers=0)
            two_marginal_logs, one_marginal_logs, no_marginal_logs = neural_symbolic_evaluate(0, f"NN evaluate test set ",
                                        test_dataloader, nbp, reasoner,args)
            
            log = {eval_queries[i] : [two_marginal_logs, one_marginal_logs, no_marginal_logs]}
            all_log[eval_queries[i]] = log
            if osp.exists('EFO-1_log/{}_result/{}'.format(args.task_folder.split("/")[-1], args.prefix)) == False:
                os.makedirs('EFO-1_log/{}_result/{}'.format(args.task_folder.split("/")[-1], args.prefix))
            with open('EFO-1_log/{}_result/{}/all_logging_test_0_{}.pickle'.format(args.task_folder.split("/")[-1], args.prefix, eval_formula_id[i]), 'wb') as f:
                    pickle.dump(log, f)
        with open('EFO-1_log/{}_result/{}/all_logging.pickle'.format(args.task_folder.split("/")[-1], args.prefix), 'wb') as g:
                pickle.dump(all_log, g)
    else:
        raise NotImplementedError
