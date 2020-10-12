import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import src.eval
import src.utils.ops as ops
from src.utils.ops import var_cuda, zeros_var_cuda
from tensorboardX import SummaryWriter

class LFramework(nn.Module):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None

        self.inference = not args.train
        self.run_analysis = args.run_analysis
        self.board_dir = args.board

        self.kg = kg
        self.mdl = mdl
        print('{} module created'.format(self.model))

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def run_train(self, train_data, dev_data):
        self.print_all_model_parameters()
        
        if self.board_dir is not None:
            writer = SummaryWriter(self.board_dir)
        else:
            writer = None
        
        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []

        batch_num = 0
        for epoch_id in range(self.start_epoch, self.num_epochs):
            print('Epoch {}'.format(epoch_id))

            # Update model parameters
            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            entropies = []
            top_rules_hit = [] # percentage of hitting top rules
            if self.run_analysis:
                rewards = None
                fns = None
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):

                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue
                    
                loss = self.loss(mini_batch)
                
                
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                
                self.optim.step()

                batch_losses.append(loss['print_loss'])
                
                if batch_num%10 == 0 and writer is not None:
                    if 'top_rule_hit' in loss.keys():
                        writer.add_scalar('data/top_rule_hit', float(loss['top_rule_hit']), batch_num)
                    if 'reward' in loss.keys():
                        writer.add_scalar('data/reward', float(torch.mean(loss['reward'])), batch_num)
                batch_num += 1
                
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                if 'top_rule_hit' in loss:
                    top_rules_hit.append(loss['top_rule_hit'])
                if self.run_analysis:
                    if rewards is None:
                        rewards = loss['reward']
                    else:
                        rewards = torch.cat([rewards, loss['reward']])
                    if fns is None:
                        fns = loss['fn']
                    else:
                        fns = torch.cat([fns, loss['fn']])
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
            if entropies:
                stdout_msg += ' entropy = {}'.format(np.mean(entropies))
            if 'top_rule_hit' in loss:
                stdout_msg += ' top rules percentage = {}'.format(np.mean(top_rules_hit))
            print(stdout_msg)
            self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                hit_ratio = num_hits / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                num_fns = float(fns.sum())
                fn_ratio = num_fns / len(fns)
                print('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
                self.eval()
                self.batch_size = self.dev_batch_size
                dev_scores, rule_scores = self.forward(dev_data, verbose=False)
                print('Dev set performance: (correct evaluation)')
                hit1, hit3, hit5, hit10, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
                # tensorboard
                if writer is not None:
                    writer.add_scalar('data/hit1_dev_correct', hit1, epoch_id)
                    writer.add_scalar('data/hit3_dev_correct', hit3, epoch_id)
                    writer.add_scalar('data/hit5_dev_correct', hit5, epoch_id)
                    writer.add_scalar('data/hit10_dev_correct', hit10, epoch_id)
                    writer.add_scalar('data/mrr_dev_correct', mrr, epoch_id)
                    if rule_scores is not None:
                        writer.add_scalar('data/confidence_score_dev', torch.mean(rule_scores).cpu().numpy(), epoch_id)
                
                # metrics = mrr
                metrics = hit1
                if self.kg.args.pre_train:
                    metrics = torch.mean(rule_scores).cpu().numpy()
                if rule_scores is not None:
                    print('average dev set rule confidence score: {}'.format(torch.mean(rule_scores).cpu().numpy()))
                print('Dev set performance: (include test set labels)')
                hit1, hit3, hit5, hit10, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
                # tensorboard
                if writer is not None:
                    writer.add_scalar('data/hit1_dev_test_label', hit1, epoch_id)
                    writer.add_scalar('data/hit3_dev_test_label', hit3, epoch_id)
                    writer.add_scalar('data/hit5_dev_test_label', hit5, epoch_id)
                    writer.add_scalar('data/hit10_dev_test_label', hit10, epoch_id)
                    writer.add_scalar('data/board/mrr_dev_test_label', mrr, epoch_id)
                
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(
                            dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)
                if self.run_analysis:
                    num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                    dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                    hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                    fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                    if epoch_id == 0:
                        with open(num_path_types_file, 'w') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'w') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'w') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))
                    else:
                        with open(num_path_types_file, 'a') as o_f:
                            o_f.write('{}\n'.format(self.num_path_types))
                        with open(dev_metrics_file, 'a') as o_f:
                            o_f.write('{}\n'.format(metrics))
                        with open(hit_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(hit_ratio))
                        with open(fn_ratio_file, 'a') as o_f:
                            o_f.write('{}\n'.format(fn_ratio))

    def forward(self, examples, verbose=False):
        pred_scores = []
        rule_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            ret = self.predict(mini_batch, verbose=verbose)
            if type(ret) is tuple:
                (pred_score, rule_score) = ret
            else:
                pred_score = ret
                rule_scores = None
            pred_scores.append(pred_score[:mini_batch_size])
            if rule_scores is not None:
                rule_scores.append(rule_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        if rule_scores is not None:
            rule_scores = torch.cat(rule_scores)
        return scores, rule_scores

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """

        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(self.model_dir, 'model_best.tar')
            shutil.copyfile(out_tar, best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            print('=> saving checkpoint to \'{}\''.format(out_tar))

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            print(r, '{}'.format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
