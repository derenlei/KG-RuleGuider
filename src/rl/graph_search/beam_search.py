import torch
import pickle
import os
import numpy as np

import src.utils.ops as ops
from src.utils.ops import unique_max, var_cuda, zeros_var_cuda, int_var_cuda, int_fill_var_cuda, var_to_numpy



def beam_search(pn, e_s, q, e_t, kg, num_steps, beam_size, return_path_components=False):
    """
    Beam search from source.

    :param pn: Policy network.
    :param e_s: (Variable:batch) source entity indices.
    :param q: (Variable:batch) query relation indices.
    :param e_t: (Variable:batch) target entity indices.
    :param kg: Knowledge graph environment.
    :param num_steps: Number of search steps.
    :param beam_size: Beam size used in search.
    :param return_path_components: If set, return all path components at the end of search.
    """
    assert (num_steps >= 1)
    batch_size = len(e_s)

    def top_k_action_r(log_action_dist):
        """
        Get top k relations.
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*k, action_space_size]
        :return:
            next_r, log_action_prob, action_offset: [batch_size*new_k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)

        action_space_size = log_action_dist.size()[1]
        # => [batch_size, k'*action_space_size]
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        k = min(beam_size, beam_action_space_size)
        # [batch_size, k]
        log_action_prob, action_ind = torch.topk(log_action_dist, k)
        next_r = (action_ind % action_space_size).view(-1)
        # [batch_size, k] => [batch_size*k]
        log_action_prob = log_action_prob.view(-1)
        # compute parent offset
        # [batch_size, k]
        action_beam_offset = action_ind / action_space_size
        # [batch_size, 1]
        action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k).unsqueeze(1)
        # [batch_size, k] => [batch_size*k]
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return next_r, log_action_prob, action_offset

    def top_k_action(log_action_dist, action_space):
        """
        Get top k actions.
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*k, action_space_size]
        :param action_space (r_space, e_space):
            r_space: [batch_size*k, action_space_size]
            e_space: [batch_size*k, action_space_size]
        :return:
            (next_r, next_e), log_action_prob, action_offset: [batch_size*new_k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)

        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]
        # => [batch_size, k'*action_space_size]
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        k = min(beam_size, beam_action_space_size)
        # [batch_size, k]
        log_action_prob, action_ind = torch.topk(log_action_dist, k)
        next_r = ops.batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
        next_e = ops.batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)
        # [batch_size, k] => [batch_size*k]
        log_action_prob = log_action_prob.view(-1)
        # compute parent offset
        # [batch_size, k]
        action_beam_offset = action_ind / action_space_size
        # [batch_size, 1]
        action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k).unsqueeze(1)
        # [batch_size, k] => [batch_size*k]
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return (next_r, next_e), log_action_prob, action_offset

    def top_k_answer_unique(log_action_dist, action_space):
        """
        Get top k unique entities
            - k = beam_size if the beam size is smaller than or equal to the beam action space size
            - k = beam_action_space_size otherwise
        :param log_action_dist: [batch_size*beam_size, action_space_size]
        :param action_space (r_space, e_space):
            r_space: [batch_size*beam_size, action_space_size]
            e_space: [batch_size*beam_size, action_space_size]
        :return:
            (next_r, next_e), log_action_prob, action_offset: [batch_size*k]
        """
        full_size = len(log_action_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        log_action_dist = log_action_dist.view(batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        log_action_prob_list = []
        action_offset_list = []
        for i in range(batch_size):
            log_action_dist_b = log_action_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()))
            unique_log_action_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = top_unique_idx / action_space_size
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = ops.pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
        next_e = ops.pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
        log_action_prob = ops.pad_and_cat(log_action_prob_list, padding_value=-ops.HUGE_INT)
        action_offset = ops.pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), log_action_prob.view(-1), action_offset.view(-1)

    def adjust_search_trace(search_trace, action_offset):
        for i, (r, e) in enumerate(search_trace):
            new_r = r[action_offset]
            new_e = e[action_offset]
            search_trace[i] = (new_r, new_e)

    def to_one_hot(x):
        y = torch.eye(kg.num_relations).cuda()
        return y[x]

    def adjust_relation_trace(relation_trace, action_offset, relation):
        return torch.cat([relation_trace[action_offset], relation.unsqueeze(1)], 1)

    # Initialization
    r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
    seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
    init_action = (r_s, e_s)
    
    # record original q
    init_q=q
    
    # path encoder
    pn.initialize_path(init_action, kg)
    if kg.args.save_beam_search_paths:
        search_trace = [(r_s, e_s)]
    relation_trace = r_s.unsqueeze(1)

    # Run beam search for num_steps
    # [batch_size*k], k=1
    log_action_prob = zeros_var_cuda(batch_size)
    if return_path_components:
        log_action_probs = []

    action = init_action
    
    # pretrain evaluation without traversing in the graph
    if kg.args.pretrain and kg.args.pretrain_out_of_graph:
        for t in range(num_steps):
            last_r, e = action
            assert (q.size() == e_s.size())
            assert (q.size() == e_t.size())
            assert (e.size()[0] % batch_size == 0)
            assert (q.size()[0] % batch_size == 0)
            k = int(e.size()[0] / batch_size)
            # => [batch_size*k]
            q = ops.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
            e_s = ops.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
            e_t = ops.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, None]

            # one step forward in relation search
            r_prob, _ = pn.transit_r(e, obs, kg)

            # => [batch_size*k, relation_space_size]
            log_action_dist = log_action_prob.view(-1, 1) + ops.safe_log(r_prob)
            #action_space = torch.stack([torch.arange(kg.num_relations)] * len(r_prob), 0).long().cuda()

            action_r, log_action_prob, action_offset = top_k_action_r(log_action_dist)

            if return_path_components:
                ops.rearrange_vector_list(log_action_probs, action_offset)
                log_action_probs.append(log_action_prob)
            pn.update_path_r(action_r, kg, offset=action_offset)

            if kg.args.save_beam_search_paths:
                adjust_search_trace(search_trace, action_offset)
            relation_trace = adjust_relation_trace(relation_trace, action_offset, action_r)

            # one step forward in entity search
            k = int(action_r.size()[0] / batch_size)
            # => [batch_size*k]
            q = ops.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
            e_s = ops.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
            e_t = ops.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
            e = e[action_offset]
            action = (action_r, e)

            seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)
            if kg.args.save_beam_search_paths:
                search_trace.append(action)


        output_beam_size = int(action[0].size()[0] / batch_size)
        # [batch_size*beam_size] => [batch_size, beam_size]
        beam_search_output = dict()
        beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
        beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)
        if kg.args.save_beam_search_paths:
            beam_search_output['search_traces'] = search_trace
        rule_score = torch.zeros(batch_size, output_beam_size).cuda().float()
        top_rules = pickle.load(open(kg.args.rule, 'rb'))
        for i in range(batch_size):
            for j in range(output_beam_size):
                path_ij = relation_trace[i * output_beam_size + j, 1 : ].cpu().numpy().tolist()
                if not int(init_q[i]) in top_rules.keys():
                    rule_score[i][j] = 0.0
                elif tuple(path_ij) in top_rules[int(init_q[i])].keys():
                    rule_score[i][j] = top_rules[int(init_q[i])][tuple(path_ij)]
                #print(tuple(relation_trace[i * output_beam_size + j, 1 : ]))
                #print(top_rules[int(init_q[i])].keys())
        # rule_score = (rule_score > 0).float()
        beam_search_output["rule_score"] = torch.mean(rule_score, 1)
        assert len(beam_search_output["rule_score"]) == batch_size
        return beam_search_output
    
    for t in range(num_steps):
        last_r, e = action
        assert (q.size() == e_s.size())
        assert (q.size() == e_t.size())
        assert (e.size()[0] % batch_size == 0)
        assert (q.size()[0] % batch_size == 0)
        k = int(e.size()[0] / batch_size)
        # => [batch_size*k]
        q = ops.tile_along_beam(q.view(batch_size, -1)[:, 0], k)
        e_s = ops.tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
        e_t = ops.tile_along_beam(e_t.view(batch_size, -1)[:, 0], k)
        obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
        # one step forward in search
        db_outcomes, _, _ = pn.transit(
            e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=True)
        action_space, action_dist = db_outcomes[0]
        
        # incorporate r_prob
        (r_space, e_space), action_mask = action_space
        r_prob, _ = pn.transit_r(e, obs, kg)
        if kg.args.pretrain:
            r_space = torch.ones(len(r_space), kg.num_relations).long() * torch.arange(kg.num_relations)
            r_space = r_space.cuda()
            e_space = torch.ones(len(e_space), kg.num_relations).long().cuda()
            action_dist = r_prob
            action_dist = action_dist * kg.r_prob_mask + (1 - kg.r_prob_mask) * ops.EPSILON
            action_space = (r_space, e_space), action_mask
        else:
            # r_space_dist = torch.matmul(r_prob.unsqueeze(1), to_one_hot(r_space).transpose(1, 2))
            # action_dist = torch.mul(r_space_dist.squeeze(1), action_dist)
            r_space_dist = torch.gather(r_prob, 1, r_space)
            action_dist = torch.mul(r_space_dist, action_dist)
            action_dist = action_dist * action_mask + (1 - action_mask) * ops.EPSILON
                    
                    
        # => [batch_size*k, action_space_size]
        log_action_dist = log_action_prob.view(-1, 1) + ops.safe_log(action_dist)
        # [batch_size*k, action_space_size] => [batch_size*new_k]
        if t == num_steps - 1 and not kg.args.pretrain:
            action, log_action_prob, action_offset = top_k_answer_unique(log_action_dist, action_space)
        else:
            action, log_action_prob, action_offset = top_k_action(log_action_dist, action_space)
        if return_path_components:
            ops.rearrange_vector_list(log_action_probs, action_offset)
            log_action_probs.append(log_action_prob)
        pn.update_path(action, kg, offset=action_offset)
        seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)
        if kg.args.save_beam_search_paths:
            adjust_search_trace(search_trace, action_offset)
            search_trace.append(action)
        relation_trace = adjust_relation_trace(relation_trace, action_offset, action[0])
    
    
    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_e2_scores'] = log_action_prob.view(batch_size, -1)
    if kg.args.save_beam_search_paths:
        beam_search_output['search_traces'] = search_trace
    
    
    rule_score = torch.zeros(batch_size, output_beam_size).cuda().float()
    top_rules = pickle.load(open(kg.args.rule, 'rb'))
    for i in range(batch_size):
        for j in range(output_beam_size):
            path_ij = relation_trace[i * output_beam_size + j, 1 : ].cpu().numpy().tolist()
            if not int(init_q[i]) in top_rules.keys():
                rule_score[i][j] = 0.0
            elif tuple(path_ij) in top_rules[int(init_q[i])].keys():
                rule_score[i][j] = top_rules[int(init_q[i])][tuple(path_ij)]
            #print(tuple(relation_trace[i * output_beam_size + j, 1 : ]))
            #print(top_rules[int(init_q[i])].keys())
    # rule_score = (rule_score > 0).float()
    beam_search_output["rule_score"] = torch.mean(rule_score, 1)
    assert len(beam_search_output["rule_score"]) == batch_size


    
    
    
    if return_path_components:
        path_width = 10
        path_components_list = []
        for i in range(batch_size):
            p_c = []
            for k, log_action_prob in enumerate(log_action_probs):
                top_k_edge_labels = []
                for j in range(min(output_beam_size, path_width)):
                    ind = i * output_beam_size + j
                    r = kg.id2relation[int(search_trace[k + 1][0][ind])]
                    e = kg.id2entity[int(search_trace[k + 1][1][ind])]
                    if r.endswith('_inv'):
                        edge_label = ' <-{}- {} {}'.format(r[:-4], e, float(log_action_probs[k][ind]))
                    else:
                        edge_label = ' -{}-> {} {}'.format(r, e, float(log_action_probs[k][ind]))
                    top_k_edge_labels.append(edge_label)
                top_k_action_prob = log_action_prob[:path_width]
                e_name = kg.id2entity[int(search_trace[1][0][i * output_beam_size])] if k == 0 else ''
                p_c.append((e_name, top_k_edge_labels, var_to_numpy(top_k_action_prob)))
            path_components_list.append(p_c)
        beam_search_output['path_components_list'] = path_components_list
    return beam_search_output
