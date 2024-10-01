from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from src.language.foq import EFO1Query
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.pipeline.reasoner import Reasoner


class NeuralSymbolicMPWithoutDynamicPruningLayer(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, nbp: NeuralBinaryPredicate, mat=None, alpha=100):
        super(NeuralSymbolicMPWithoutDynamicPruningLayer, self).__init__()
        self.nbp = nbp
        self.feature_dim = nbp.entity_embedding.size(1)
        self.num_entities = nbp.num_entities
        
        transposed_mats = []
        for i in range(len(mat)):
            transposed_mats.append(mat[i].t().coalesce())
        self.register_buffer('mat', torch.stack(transposed_mats))
        
        if alpha > 10000:
            self.alpha = nbp.num_entities
        else:
            self.alpha = alpha

        # Initialization with zeros can be regarded as an approximation of dynamic pruning, 
        # because the all-0 vector will be converted to a uniform fuzzy vector with low probability after softmax. 
        # In this case, most of the noise is filtered. Therefore, random initialization should be adopted when considering variants without dynamic pruning.
        self.existential_embedding = torch.rand((1, self.feature_dim)).to(self.nbp.device)
        self.universal_embedding = torch.rand((1, self.feature_dim)).to(self.nbp.device)
        self.free_embedding = torch.rand((1, self.feature_dim)).to(self.nbp.device)

        self.clip_para = torch.Tensor([1e-14]).to(self.nbp.device)

    def clip_norm(self, vector):
        vector_norm = vector.masked_fill(vector < self.clip_para, 0) / torch.max(self.clip_para, torch.sum(vector, dim=-1).unsqueeze(-1))
        return vector_norm

    def fuzzy_vector_to_embedding(self, fuzzy_vector): 
        embedding = fuzzy_vector @ self.nbp.entity_embedding
        return embedding

    def neural_enhanced_symbolic_reasoning(self, symbolic_vector, neural_embedding):
        score = self.nbp.kge_similarity(self.nbp.entity_embedding.unsqueeze(0), neural_embedding.unsqueeze(1))
        neural_vector = torch.softmax(score, dim=-1)
        vector_enhanced = symbolic_vector + neural_vector
        vector_enhanced = self.clip_norm(vector_enhanced)
        return vector_enhanced
    
    def product_conjunction(self, fuzzy_vector_list):
        result = fuzzy_vector_list[0]
        for i in range(1, len(fuzzy_vector_list)):
            result = result * fuzzy_vector_list[i]

        result = self.clip_norm(result)
        return result

    def product_disjunction(self, fuzzy_vector1, fuzzy_vector2):
        result = self.clip_norm(fuzzy_vector1 + fuzzy_vector2 - fuzzy_vector1 * fuzzy_vector2)
        return result

    def message_passing(self,
                        term_emb_dict,
                        atomic_dict,
                        pred_emb_dict,
                        inv_pred_emb_dict,
                        term_vector_dict,
                        pred_relation_idx_dict,
                        inv_pred_relation_idx_dict
                        ):
        
        term_collect_vectors_dict = defaultdict(list)
        
        for predicate, atomic in atomic_dict.items():
            head_name, tail_name = atomic.head.name, atomic.tail.name
            
            head_emb = term_emb_dict[head_name]
            head_vector = term_vector_dict[head_name]
            
            tail_emb = term_emb_dict[tail_name]
            tail_vector = term_vector_dict[tail_name]
            
            sign = -1 if atomic.negated else 1
            pred_emb = pred_emb_dict[atomic.relation]
            
            if head_emb.size(0) == 1:
                head_emb = head_emb.expand(pred_emb.size(0), -1)
                head_vector = head_vector.expand(pred_emb.size(0), -1)
            if tail_emb.size(0) == 1:
                tail_emb = tail_emb.expand(pred_emb.size(0), -1)
                tail_vector = tail_vector.expand(pred_emb.size(0), -1)

            assert head_emb.size(0) == pred_emb.size(0)
            assert tail_emb.size(0) == pred_emb.size(0)

            if tail_name in ['f', 'e1', 'e2', 'e3', 'f1', 'f2']: # TODO
                neural_tail_embedding = sign * self.nbp.estimate_tail_emb(head_emb, pred_emb)
                
                symbolic_tail_vector = torch.stack([torch.sparse.mm(self.mat[pred_relation_idx_dict[atomic.relation][i]], head_vector[i].unsqueeze(1)).squeeze(1)
                                        for i in range(len(pred_relation_idx_dict[atomic.relation]))])  
                
                if sign == -1:
                    symbolic_tail_vector = self.alpha / self.nbp.num_entities - symbolic_tail_vector 
                symbolic_tail_vector = self.clip_norm(symbolic_tail_vector)
                symbolic_tail_vector_enhanced = self.neural_enhanced_symbolic_reasoning(symbolic_tail_vector, neural_tail_embedding)

                term_collect_vectors_dict[tail_name].append(symbolic_tail_vector_enhanced)
            else:
                term_collect_vectors_dict[tail_name] = None

            if head_name in ['f', 'e1', 'e2', 'e3', 'f1', 'f2']:  # TODO
                neural_head_embedding = sign * self.nbp.estimate_head_emb(tail_emb, pred_emb)
                
                symbolic_head_vector = torch.stack([torch.sparse.mm(self.mat[inv_pred_relation_idx_dict[atomic.relation][i]], tail_vector[i].unsqueeze(1)).squeeze(1)
                                        for i in range(len(inv_pred_relation_idx_dict[atomic.relation]))])  
                
                if sign == -1:
                    symbolic_head_vector = self.alpha / self.nbp.num_entities - symbolic_head_vector
                symbolic_head_vector = self.clip_norm(symbolic_head_vector)
                symbolic_head_vector_enhanced = self.neural_enhanced_symbolic_reasoning(symbolic_head_vector, neural_head_embedding)

                term_collect_vectors_dict[head_name].append(symbolic_head_vector_enhanced)
            else:
                term_collect_vectors_dict[head_name] = None

        return term_collect_vectors_dict

    def forward(self, 
                init_term_emb_dict, 
                predicates, 
                pred_emb_dict, 
                inv_pred_emb_dict,
                init_term_vector_dict,
                pred_relation_idx_dict,
                inv_pred_relation_idx_dict
                ):
        
        term_collect_vectors_dict = self.message_passing(
            init_term_emb_dict,
            predicates,
            pred_emb_dict,
            inv_pred_emb_dict,
            init_term_vector_dict,
            pred_relation_idx_dict,
            inv_pred_relation_idx_dict
        )

        out_term_emb_dict = {}
        out_term_vector_dict = {}

        for t, collect_vector_list in term_collect_vectors_dict.items():
            if t not in ['f', 'e1', 'e2', 'e3', 'f1', 'f2']: 
                out_term_vector_dict[t] = init_term_vector_dict[t]
                continue

            agg_vec = self.product_conjunction(collect_vector_list)
            out_term_vector_dict[t] = agg_vec
        
        for t, _ in term_collect_vectors_dict.items():
            if t not in ['f', 'e1', 'e2', 'e3', 'f1', 'f2']:  
                out_term_emb_dict[t] = init_term_emb_dict[t]
                continue
            
            agg_emb = self.fuzzy_vector_to_embedding(out_term_vector_dict[t])
            out_term_emb_dict[t] = agg_emb
            
        return out_term_emb_dict, out_term_vector_dict


class NeuralSymbolicMPWithoutDynamicPruningReasoner(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 lgnn_layer: NeuralSymbolicMPWithoutDynamicPruningLayer,
                 depth_shift=0
                 ):
        self.nbp = nbp
        self.lgnn_layer = lgnn_layer        # formula dependent
        self.depth_shift = depth_shift
        self.formula: EFO1Query = None
        self.term_local_emb_dict = {}
        self.term_local_vector_dict = {}
        self.query_type = None  # TODO

    def initialize_with_query(self, formula, query_type):
        self.formula = formula
        self.query_type = query_type 
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}

    def initialize_local_embedding(self):
        term_vector_dict = {}
        for term_name in self.formula.term_dict:
            if self.formula.has_term_grounded_entity_id_list(term_name):
                entity_id = self.formula.get_term_grounded_entity_id_list(term_name)
                emb = self.nbp.get_entity_emb(entity_id)
                vector = F.one_hot(torch.tensor(entity_id, dtype=torch.int64).to(self.nbp.device), num_classes=self.nbp.num_entities).float()
            elif self.formula.term_dict[term_name].is_existential:
                emb = self.lgnn_layer.existential_embedding
                vector = torch.zeros(1, self.nbp.num_entities).to(self.nbp.device) 
            elif self.formula.term_dict[term_name].is_free:
                emb = self.lgnn_layer.free_embedding
                vector = torch.zeros(1, self.nbp.num_entities).to(self.nbp.device)
            elif self.formula.term_dict[term_name].is_universal:
                emb = self.lgnn_layer.universal_embedding
                vector = torch.zeros(1, self.nbp.num_entities).to(self.nbp.device)
            else:
                raise KeyError(f"term name {term_name} cannot be initialized")
            self.set_local_embedding(term_name, emb)
            term_vector_dict[term_name] = vector
        return term_vector_dict

    def estimate_variable_embeddings(self):
        term_vector_dict = self.initialize_local_embedding() # tensor
        term_emb_dict = self.term_local_emb_dict
        pred_emb_dict = {}
        inv_pred_emb_dict = {}
        pred_relation_idx_dict = {} # tensor
        inv_pred_relation_idx_dict = {} # tensor
        for atomic_name in self.formula.atomic_dict:
            pred_name = self.formula.atomic_dict[atomic_name].relation
            if self.formula.has_pred_grounded_relation_id_list(pred_name):
                pred_emb_dict[pred_name], pred_relation_idx_dict[pred_name] = self.get_rel_emb(pred_name)
                inv_pred_emb_dict[pred_name], inv_pred_relation_idx_dict[pred_name] = self.get_rel_emb(pred_name, inv=True)

        for _ in range(
            max(1, self.formula.quantifier_rank + self.depth_shift)
        ):
            term_emb_dict, term_vector_dict = self.lgnn_layer(
                term_emb_dict,
                self.formula.atomic_dict,
                pred_emb_dict,
                inv_pred_emb_dict,
                term_vector_dict,
                pred_relation_idx_dict,
                inv_pred_relation_idx_dict
            )

        for term_name in term_emb_dict:
            self.term_local_emb_dict[term_name] = term_emb_dict[term_name]
            self.term_local_vector_dict[term_name] = term_vector_dict[term_name]
