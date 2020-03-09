# Copyright 2020 University of Toronto, all rights reserved

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        # initialize parameterized submodules here: rnn, embedding
        # using: self.source_vocab_size, self.word_embedding_size, self.pad_id,
        # self.dropout, self.cell_type, self.hidden_state_size,
        # self.num_hidden_layers
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        # torch.nn.{LSTM, GRU, RNN, Embedding}
        self.embedding = torch.nn.Embedding(self.source_vocab_size, 
                                            self.word_embedding_size, 
                                            padding_idx = self.pad_id)
        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(self.word_embedding_size, self.hidden_state_size, 
                                     self.num_hidden_layers, dropout=self.dropout, 
                                     bidirectional=True)
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(self.word_embedding_size, self.hidden_state_size, 
                                    self.num_hidden_layers, dropout=self.dropout, 
                                    bidirectional=True)
        else:
            self.rnn = torch.nn.RNN(self.word_embedding_size, self.hidden_state_size, 
                                    self.num_hidden_layers, dropout=self.dropout, 
                                    bidirectional=True)


    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        x = self.embedding(F)
        return x
        

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # compute all final hidden states for provided input sequence.
        # make sure you handle padding properly!
        # x is of shape (S, N, I)
        # F_lens is of shape (N,)
        # h_pad is a float
        # h (output) is of shape (S, N, 2 * H)
        # relevant pytorch modules:
        # torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted = False)
        output, hidden = self.rnn(x)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value = h_pad)
        return h

class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # initialize parameterized submodules: embedding, cell, ff
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # relevant pytorch modules:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # torch.nn.{Embedding,Linear,LSTMCell,RNNCell,GRUCell}
      
        self.embedding = torch.nn.Embedding(self.target_vocab_size, 
                                            self.word_embedding_size, self.pad_id)
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size, self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size, self.hidden_state_size)
        else:
            self.cell = torch.nn.RNNCell(self.word_embedding_size, self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each
        # direction:
        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        # relevant pytorch modules: torch.cat

        temp = torch.arange(len(F_lens))
        h_f = h[F_lens-1, temp, :self.hidden_state_size//2]
        h_b = h[0, temp, self.hidden_state_size // 2:]
        # exit()
    
        htilde_tm1 = torch.cat([h_f, h_b], dim =1) 
        return htilde_tm1

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
     
        xtilde_t = self.embedding(E_tm1)

        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
       
        return self.cell(xtilde_t, htilde_tm1)

    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        return self.ff(htilde_t)


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        self.embedding = torch.nn.Embedding(self.target_vocab_size, 
                                            self.word_embedding_size, self.pad_id)
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size + 
                                          self.hidden_state_size, self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size +
                                         self.hidden_state_size, self.hidden_state_size)
        else:
            self.cell = torch.nn.RNNCell(self.word_embedding_size + 
                                         self.hidden_state_size, self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        return torch.zeros_like(h[0], device=h.device)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        xtilde_t = torch.cat([self.embedding(E_tm1), self.attend(htilde_tm1, h, F_lens)], dim = 1)
        # print("xtilde_t:", xtilde_t.size())
        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        
        c_t = torch.zeros_like(h[0], device = h.device)
 
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)

        for i in range(len(F_lens)):
            A = alpha_t[:,i].unsqueeze(1)
            v = torch.mm(A.t(), h[:,i,:])
            c_t[i] = v
        
        return c_t

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        S = len(h)
        N = len(htilde_t)
        e_t = torch.empty(size=(S, N))
        for i in range(len(h)):
            e = torch.nn.functional.cosine_similarity(htilde_t, h[i])
            e_t[i] = e.clone()
        # print("e_t: ", e_t.size())
        # exit()
        return e_t


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        # self.target_vocab_size, self.target_eos
        # Recall that self.target_eos doubles as the decoder pad id since we
        # never need an embedding for it
        self.encoder = encoder_class(self.source_vocab_size, self.source_pad_id, 
                                     self.word_embedding_size, self.encoder_num_hidden_layers,
                                     self.encoder_hidden_size, self.encoder_dropout, self.cell_type)
    
        self.decoder = decoder_class(self.target_vocab_size, self.target_eos, 
                                     self.word_embedding_size, self.encoder_hidden_size*2, 
                                     self.cell_type)

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the
        # sequence.
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # E is of shape (T, N)
        # logits (output) is of shape (T - 1, N, Vo)
        # relevant pytorch modules: torch.{zero_like,stack}
        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)
        T, N = E.size()
        Vo = self.target_vocab_size
        logits = torch.empty(size=(T-1, N, Vo))
        htilde_t = None
        # htilde_t = self.decoder.get_first_hidden_state(h, F_lens)
        # print("E[0]: ", len(E[0]))
        # print("E: ", len(E))
        for t in range(len(E)-1):
            # E_clone = E[t].clone()
            # xtilde_t = self.decoder.get_current_rnn_input(E_clone, htilde_t, h, F_lens)
            # htilde_t = self.decoder.get_current_hidden_state(xtilde_t, htilde_t)
            # logits[t] = self.decoder.get_current_logits(htilde_t)

            if htilde_t is None:
                htilde_t = self.decoder.get_first_hidden_state(h, F_lens)
                # print("htilde_t: ", htilde_t.size())
                if self.cell_type == 'lstm':
                    # initialize cell state with zeros
                    htilde_t = (htilde_t, torch.zeros_like(htilde_t))
            # fixed this, was cloning each row instead of each column 
            E_clone = E[t].clone()
            # print("E_clone: ", E_clone.size())
            # print("F_lens: ", F_lens.size())
            # print("h: ", h.size())
          
            xtilde_t = self.decoder.get_current_rnn_input(E_clone, htilde_t, h, F_lens)
            # print("xtilde_t: ", xtilde_t.size())
            htilde_t = self.decoder.get_current_hidden_state(xtilde_t, htilde_t)
            if self.cell_type == 'lstm':
                logits[t] = self.decoder.get_current_logits(htilde_t[0])
            else:
                logits[t] = self.decoder.get_current_logits(htilde_t)

        return logits
        
        
    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of
        #                                                         those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)
        # relevant pytorch modules:
        # torch.{flatten,topk,unsqueeze,expand_as,gather,cat}
        # hint: if you flatten a two-dimensional array of shape z of (A, B),
        # then the element z[a, b] maps to z'[a*B + b]

        assert self.beam_width == 1, "Greedy requires beam width of 1"
        extensions_t = (logpb_tm1.unsqueeze(-1) + logpy_t).squeeze(1)  # (N, V)
        logpb_t, v = extensions_t.max(1)  # (N,), (N,)
        logpb_t = logpb_t.unsqueeze(-1)  # (N, 1) == (N, K)
        # v indexes the maximal element in dim=1 of extensions_t that was
        # chosen, which equals the token index v in k -> v
        v = v.unsqueeze(0).unsqueeze(-1)  # (1, N, 1) == (1, N, K)
        b_t_1 = torch.cat([b_tm1_1, v], dim=0)
        # For greedy search, all paths come from the same prefix, so
        b_t_0 = htilde_t
        return b_t_0, b_t_1, logpb_t
        
