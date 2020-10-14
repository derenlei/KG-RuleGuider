import pickle
import torch
import numpy as np
import src.rl.graph_search.beam_search as search
import src.utils.ops as ops
from src.learn_framework import LFramework
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval

        # Inference hyperparameters
        self.beam_size = args.beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

        self.rel2rules = pickle.load(open(args.rule, 'rb'))

        self.r_rule = args.rule_ratio
        if args.pretrain:
            self.r_rule = 1.0
        self.pretrain = args.pretrain
        self.pretrain_out_of_graph = args.pretrain_out_of_graph
        self.pretrain_teacher_forcing = args.pretrain_teacher_forcing
        
        self.r_prob_mask = kg.r_prob_mask

    def reward_fun(self, e1, r, e2, pred_e2):
        return (pred_e2 == e2).float()#, (pred_e2 == e2)

    def rule_reward_fun(self, r, path_trace, hit_reward_binary):
        if self.pretrain and self.pretrain_out_of_graph:
            path = torch.stack(path_trace[1:], 1).cpu().numpy().tolist()
        else:
            path = []
            for i, (rel, ent) in enumerate(path_trace):
                if i > 0:
                    path.append(rel.unsqueeze(1))
            path = torch.cat(path, 1).cpu().numpy().tolist()
        assert len(path) == len(r)
        reward = torch.zeros(r.shape).cuda().float()
        for i in range(len(path)):
            if int(r[i]) in self.rel2rules.keys() and tuple(path[i]) in self.rel2rules[int(r[i])].keys():
                reward[i] = self.rel2rules[int(r[i])][tuple(path[i])]
        return reward

    def loss(self, mini_batch):
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r

        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        
        if self.pretrain and self.pretrain_out_of_graph:
            if self.pretrain_teacher_forcing:
                output = self.teacher_forcing_pretrain(e1, r, e2, num_steps=self.num_rollout_steps)
            else:
                output = self.rollout_pretrain(e1, r, e2, num_steps=self.num_rollout_steps)
            log_action_probs = output['log_action_probs']
            action_entropy = output['action_entropy']
            rule_reward = self.rule_reward_fun(r, output['path_trace'], None)
            final_reward = rule_reward
        else:
            output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

            # Compute policy gradient loss
            pred_e2 = output['pred_e2']
            log_action_probs = output['log_action_probs']
            action_entropy = output['action_entropy']

            # Compute discounted reward
            r_rule = self.r_rule # ratio of rule reward
            hit_reward, hit_reward_binary = self.reward_fun(e1, r, e2, pred_e2)
            rule_reward = self.rule_reward_fun(r, output['path_trace'], hit_reward_binary)

            final_reward = (1 - r_rule) * hit_reward + r_rule * rule_reward
        
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        
        top_rule_hit = rule_reward.cpu().numpy()
        top_rule_hit = np.where(top_rule_hit>0,1,0)
        loss_dict['top_rule_hit'] = top_rule_hit.mean() # percentage of hitting top rules
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    def teacher_forcing_pretrain(self, e_s, q, e_t, num_steps):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :return log_action_probs: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []

        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        path_label = []
        cnt = 0
        for q_b in q:
            if int(q_b) not in self.rel2rules.keys():
                path_label.append(torch.randint(kg.num_relations, (num_steps,)).numpy().tolist())
                continue
            rules = self.rel2rules[int(q_b)]
            cnt += 1
            # uniform
            # sample_rule_id = torch.randint(len(rules.keys()), (1,)).item()
            
            # weighted by confidence score
            rule_dist_orig = torch.tensor(list(rules.values())).cuda()
            rand = torch.rand(rule_dist_orig.size())
            keep_mask = var_cuda(rand > self.action_dropout_rate).float()
            rule_dist = rule_dist_orig * keep_mask
            rule_sum = torch.sum(rule_dist, 0)
            is_zero = (rule_sum == 0).float()#.unsqueeze(1)
            rule_dist = rule_dist + is_zero * rule_dist_orig
            sample_rule_id = torch.multinomial(rule_dist, 1).item()
            path_label.append(list(rules.keys())[sample_rule_id])
        path_label = torch.tensor(path_label).cuda()
        #print('rule_path_percentage:', cnt/len(path_label))

        path_trace = [r_s]
        pn.initialize_path((r_s, e_s), kg)
        for t in range(num_steps):
            last_r = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, None]

            # relation selection
            r_prob, policy_entropy = pn.transit_r(None, obs, kg)
            action_r = path_label[:, t]
            action_prob = ops.batch_lookup(r_prob, action_r.view(-1,1))
            pn.update_path_r(action_r, kg)

            action_entropy.append(policy_entropy)

            log_action_probs.append(ops.safe_log(action_prob))
            path_trace.append(action_r)


        return {
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace
        }

    def rollout_pretrain(self, e_s, q, e_t, num_steps):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :return log_action_probs: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []

        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)

        path_trace = [r_s]
        pn.initialize_path((r_s, e_s), kg)
        for t in range(num_steps):

            last_r = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, None]

            # relation selection
            r_prob, policy_entropy = pn.transit_r(None, obs, kg)
            sample_outcome_r = self.sample_relation(r_prob)
            action_r = sample_outcome_r['action_sample']
            action_prob = sample_outcome_r['action_prob']
            pn.update_path_r(action_r, kg)

            action_entropy.append(policy_entropy)

            log_action_probs.append(ops.safe_log(action_prob))
            path_trace.append(action_r)


        return {
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace
        }
    
    def sample_relation(self, r_dist, inv_offset=None):
        """
        Sample a relation based on current policy.
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_r: Sampled next relation.
        :return r_prob: Probability of the sampled relation.
        """

        if inv_offset is not None:
            raise NotImplementedError('Relation bucket not implemented!')
        else:
            sample_dist = r_dist
            next_r = torch.multinomial(sample_dist, 1, replacement = True)
            r_prob = ops.batch_lookup(sample_dist, next_r)
            sample_outcome = {}
            sample_outcome['action_sample'] = next_r.view(-1)
            sample_outcome['action_prob'] = r_prob.view(-1)

        return sample_outcome

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_action_probs: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []
        
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)
        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy_e = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            r_prob, policy_entropy_r = pn.transit_r(e, obs, kg)
            sample_outcome = self.sample_action(obs, t, path_trace, db_outcomes, r_prob, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy_e)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }

    def sample_action(self, obs, t, path_trace, db_outcomes, r_prob, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def find_mute_idx(space, valid_space):
            space = space.cpu().numpy()
            # action_mute_mask = np.zeros(space.shape).astype(np.float32)
            action_mute_mask = []
            # for v in valid_space:
            #     action_mute_mask += space == v
            for idx in range(len(space)):
                # if space[idx] == 0:
                #     break
                if space[idx] in valid_space:
                    #     action_mute_mask[idx] = 1
                    action_mute_mask.append(1)
                else:
                    action_mute_mask.append(0)
            return var_cuda(torch.FloatTensor(action_mute_mask))
        
        def to_one_hot(x):
            y = torch.eye(self.kg.num_relations).cuda()
            return y[x]

        def apply_action_dropout_mask(r_space, action_dist, action_mask):
            
            bucket_size = len(r_space)
            batch_idx = torch.tensor(offset[self.ct : self.ct + bucket_size]).cuda()
            r_prob_b = r_prob[batch_idx]
            
            if not self.pretrain:
                uni_mask = torch.zeros(r_prob_b.shape).cuda()
                uni_mask = torch.scatter(uni_mask, 1, r_space, torch.ones(r_space.shape).cuda()).cuda()
                r_prob_b = r_prob_b * uni_mask
            r_prob_sum = torch.sum(r_prob_b, 1)
            is_zero = (r_prob_sum == 0).float().unsqueeze(1)
            r_prob_b = r_prob_b + is_zero * torch.ones(r_prob_b[0].size()).cuda()
            
            
            r_chosen_b = torch.multinomial(r_prob_b, 1, replacement = True)
            
            
            r_prob_chosen_b = ops.batch_lookup(r_prob_b, r_chosen_b).unsqueeze(1)
            
            action_mute_mask = (r_space == r_chosen_b).float()
            if self.pretrain:
                action_dist_muted = action_mute_mask * r_prob_chosen_b
            else:
                action_dist_muted = action_dist * action_mute_mask * r_prob_chosen_b

            dist_sum = torch.sum(action_dist_muted, 1)
            is_zero = (dist_sum == 0).float().unsqueeze(1)
            if self.pretrain:
                uniform_dist = torch.ones(action_dist[0].size()).float().cuda()
                action_dist_muted = action_dist_muted + is_zero * uniform_dist
            else:
                action_dist_muted = action_dist_muted + is_zero * action_dist
            self.ct += bucket_size
            new_action_dist = action_dist_muted
            
            
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                sample_action_dist = new_action_dist * action_keep_mask + ops.EPSILON * (
                        1 - action_keep_mask) * action_mask
                # if dropout to 0, keep original value
                dist_sum = torch.sum(sample_action_dist, 1)
                is_zero = (dist_sum == 0).float().unsqueeze(1)
                sample_action_dist = sample_action_dist + is_zero * new_action_dist
                return sample_action_dist
            else:
                return new_action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(r_space, action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(sample_action_dist, idx)
            
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        offset = [0] * len(inv_offset)
        for i in range(len(inv_offset)):
            offset[inv_offset[i]] = i
        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            self.ct = 0
            self.zero_ct = 0
            
            # relation dropout
            rand = torch.rand(r_prob.size())
            r_prob_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
            r_prob = r_prob * r_prob_keep_mask
            
            r_prob_sum = torch.sum(r_prob, 1)
            is_zero = (r_prob_sum == 0).float().unsqueeze(1)
            r_prob = r_prob + is_zero * torch.ones(r_prob[0].size()).cuda()
            # r_prob_keep_mask = (rand > self.action_dropout_rate).float().cuda()
            #r_chosen = torch.multinomial(r_prob, 1, replacement = True)
            #r_prob_chosen = ops.batch_lookup(r_prob, r_chosen)
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']
        if verbose:
            # print inference paths
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                        print('beam {}: score = {} \n<PATH> {}'.format(
                            j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i].long()] = torch.exp(pred_e2_scores[i])
        return pred_scores, beam_search_output["rule_score"]

    def record_path_trace(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]
