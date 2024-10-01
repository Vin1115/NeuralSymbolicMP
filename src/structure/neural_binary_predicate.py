from abc import abstractmethod

import torch


class NeuralBinaryPredicate:
    num_entities: int
    num_relations: int
    device: torch.device
    scale: float

    @abstractmethod
    def embedding_score(self, head_emb, rel_emb, tail_emb):
        """
        This method computes the score for the triple given the head, tail and
        relation embedding. The higher score means more likely to be a predicate.
        Inputs:
            Three embeddings are in the shape [..., embed_dim]
        Returns:
            The tensor of scores in the shape [...]
        """
        pass

    @abstractmethod
    def score2truth_value(self, score):
        pass

    @abstractmethod
    def estimate_tail_emb(self, head_emb, rel_emb):
        pass

    @abstractmethod
    def estimate_head_emb(self, tail_emb, rel_emb):
        pass

    @abstractmethod
    def estiamte_rel_emb(self, head_emb, tail_emb):
        pass

    @abstractmethod
    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self._relation_embedding(rel_id)

    @abstractmethod
    def get_entity_emb(self, entity_id_or_tensor):
        pass

    @abstractmethod
    def get_random_entity_embed(self, batch_size):
        pass

    @property
    def entity_embedding(self) -> torch.Tensor:
        pass

    @property
    def relation_embedding(self) -> torch.Tensor:
        pass

    @classmethod
    def create(cls, device, **kwargs):
        obj = cls(device=device, **kwargs)
        obj = obj.to(device)
        return obj

    def batch_predicate_score(self,
                              triple_tensor: torch.Tensor) -> torch.Tensor:
        """
        This method computes the scores for the triple. triple tensors the
        shape of [..., 3]
        It returns the same size of predicate scores.
        """
        if isinstance(triple_tensor, list):
            assert len(triple_tensor) == 3
            head_id_ten, rel_id_ten, tail_id_ten = triple_tensor
        else:
            head_id_ten, rel_id_ten, tail_id_ten = torch.split(
                triple_tensor, 1, dim=-1)
        head_emb = self._entity_embedding(head_id_ten)
        rel_emb = self._relation_embedding(rel_id_ten)
        tail_emb = self._entity_embedding(tail_id_ten)
        return self.embedding_score(head_emb, rel_emb, tail_emb)

    def get_all_entity_rankings(self, batch_embedding_input, eval_batch_size=16, score='cos'):
        batch_size = batch_embedding_input.size(0)
        begin = 0
        entity_ranking_list = []
        for begin in range(0, batch_size, eval_batch_size):
            end = begin + eval_batch_size
            eval_batch_embedding_input = batch_embedding_input[begin: end]
            eval_batch_embedding_input = eval_batch_embedding_input.unsqueeze(-2)
            # batch_size, all_candidates
            # ranking score should be the higher the better
            # ranking_score[entity_id] = the score of {entity_id}
            # ranking_score = self.entity_pair_scoring(eval_batch_embedding_input, self.entity_embedding)
            if score == 'cos':
                ranking_score = torch.cosine_similarity(eval_batch_embedding_input, self.entity_embedding, dim=-1)
            else:
                ranking_score = - torch.norm(eval_batch_embedding_input - self.entity_embedding, dim=-1)
            # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
            ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
            # entity_rankings[entity_id] = {rankings} of the entity
            entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
            entity_ranking_list.append(entity_rankings)

        batch_entity_rankings = torch.cat(entity_ranking_list, dim=0)
        return batch_entity_rankings

    def get_all_entity_rankings_ns(self, batch_embedding_input, batch_vector_input, eval_batch_size=16, score='cos', neural_value=0.5, symbolic_value=0.5):
        batch_size = batch_embedding_input.size(0)
        begin = 0
        entity_ranking_list = []
        for begin in range(0, batch_size, eval_batch_size):
            end = begin + eval_batch_size
            eval_batch_embedding_input = batch_embedding_input[begin: end]
            eval_batch_embedding_input = eval_batch_embedding_input.unsqueeze(-2)
            eval_batch_vector_input = batch_vector_input[begin: end]

            ranking_score = torch.cosine_similarity(eval_batch_embedding_input, self.entity_embedding, dim=-1)

            entity_emb_logit = torch.softmax(ranking_score, dim=-1)
            fuzzy_logit = neural_value * entity_emb_logit + symbolic_value * eval_batch_vector_input
            ranked_entity_ids = torch.argsort(fuzzy_logit, dim=-1, descending=True)  
            entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
            entity_ranking_list.append(entity_rankings)

        batch_entity_rankings = torch.cat(entity_ranking_list, dim=0)

        return batch_entity_rankings

    def clip_norm(self, vector):
        clip_para = torch.Tensor([1e-14]).to(self.device)
        vector_norm = vector.masked_fill(vector < clip_para, 0) / torch.max(clip_para, torch.sum(vector, dim=-1).unsqueeze(-1))
        return vector_norm

    def get_all_entity_rankings_ns_dnf(self, batch_embedding_input1, batch_embedding_input2, batch_vector_input1,
                                       batch_vector_input2, eval_batch_size=16, score='cos', neural_value=0.5,
                                       symbolic_value=0.5):
        batch_size = batch_embedding_input1.size(0)
        begin = 0
        entity_ranking_list = []
        for begin in range(0, batch_size, eval_batch_size):
            end = begin + eval_batch_size
            eval_batch_embedding_input1 = batch_embedding_input1[begin: end]
            eval_batch_embedding_input2 = batch_embedding_input2[begin: end]
            eval_batch_embedding_input1 = eval_batch_embedding_input1.unsqueeze(-2)
            eval_batch_embedding_input2 = eval_batch_embedding_input2.unsqueeze(-2)
            eval_batch_vector_input1 = batch_vector_input1[begin: end]
            eval_batch_vector_input2 = batch_vector_input2[begin: end]

            ranking_score1 = torch.cosine_similarity(eval_batch_embedding_input1, self.entity_embedding, dim=-1)
            entity_emb_logit1 = torch.softmax(ranking_score1, dim=-1)
            fuzzy_logit1 = self.clip_norm(neural_value * entity_emb_logit1 + symbolic_value * eval_batch_vector_input1)

            ranking_score2 = torch.cosine_similarity(eval_batch_embedding_input2, self.entity_embedding, dim=-1)
            entity_emb_logit2 = torch.softmax(ranking_score2, dim=-1)
            fuzzy_logit2 = self.clip_norm(neural_value * entity_emb_logit2 + symbolic_value * eval_batch_vector_input2)

            temp = 1000
            softmax_logit1 = torch.softmax(fuzzy_logit1 * temp, dim=-1)
            softmax_logit2 = torch.softmax(fuzzy_logit2 * temp, dim=-1)

            fuzzy_logit = torch.max(softmax_logit1, softmax_logit2)

            ranked_entity_ids = torch.argsort(fuzzy_logit, dim=-1, descending=True)
            entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
            entity_ranking_list.append(entity_rankings)

        batch_entity_rankings = torch.cat(entity_ranking_list, dim=0)

        return batch_entity_rankings

    def get_all_entity_rankings_dnf11(self, batch_embedding_input1, batch_embedding_input2, batch_vector_input1,
                                      batch_vector_input2, eval_batch_size=16, score='cos', neural_value=0.5,
                                      symbolic_value=0.5):
        '''
        A bad implementation. Only sub-conjunctive query 1 is taken to compute metrics.
        '''
        batch_size = batch_embedding_input1.size(0)
        begin = 0
        entity_ranking_list = []
        for begin in range(0, batch_size, eval_batch_size):
            end = begin + eval_batch_size
            eval_batch_embedding_input1 = batch_embedding_input1[begin: end]
            eval_batch_embedding_input1 = eval_batch_embedding_input1.unsqueeze(-2)
            eval_batch_vector_input1 = batch_vector_input1[begin: end]

            ranking_score1 = torch.cosine_similarity(eval_batch_embedding_input1, self.entity_embedding, dim=-1)
            entity_emb_logit1 = torch.softmax(ranking_score1, dim=-1)
            fuzzy_logit1 = self.clip_norm(neural_value * entity_emb_logit1 + symbolic_value * eval_batch_vector_input1)

            ranked_entity_ids = torch.argsort(fuzzy_logit1, dim=-1, descending=True)
            entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
            entity_ranking_list.append(entity_rankings)

        batch_entity_rankings = torch.cat(entity_ranking_list, dim=0)

        return batch_entity_rankings

    def get_all_entity_rankings_v2(self, batch_embedding_input, eval_batch_size=16, score='cos'):
        batch_size = batch_embedding_input.size(0)
        entity_ranking_list = []
        for begin in range(0, batch_size, eval_batch_size):
            end = begin + eval_batch_size
            eval_batch_embedding_input = batch_embedding_input[begin: end]
            eval_batch_embedding_input = eval_batch_embedding_input.unsqueeze(-2)  # batch*disj_num*1*dim
            if score == 'cos':
                disjunctive_ranking_score = torch.cosine_similarity(
                    eval_batch_embedding_input, self.entity_embedding, dim=-1)
            else:
                disjunctive_ranking_score = - torch.norm(eval_batch_embedding_input - self.entity_embedding, dim=-1)
            # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
            ranking_score, _ = torch.max(disjunctive_ranking_score, dim=1)
            ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
            entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
            entity_ranking_list.append(entity_rankings)

        batch_entity_rankings = torch.cat(entity_ranking_list, dim=0)
        return batch_entity_rankings

    @abstractmethod
    def entity_pair_scoring(self, emb1, emb2):
        pass
