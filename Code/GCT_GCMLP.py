class GCT_GCMLP(nn.Module):
    def __init__(self, ntoken, encoder_layers, decoder_layers, d_model=config.d_model, embed_dim = 768):
        super(GCT_GCMLP, self).__init__()
        self.encoder = Encoder(n_layers=encoder_layers, embed_dim=embed_dim)
        self.decoder = Decoder(ntoken=ntoken, n_layers=decoder_layers)

        self.gated_projection = nn.Linear(ntoken, ntoken)
        self.projection = nn.Linear(d_model, ntoken)
        self.projection2 = nn.Linear(ntoken, ntoken)
        self.layerNorm = nn.LayerNorm(ntoken)

        self.my_init_weight()

    def my_init_weight(self):
        init_layer(self.gated_projection)
        init_layer(self.projection)
        init_layer(self.projection2)

    def forward(self, enc_inputs, dec_inputs, reverse_dec_inputs, using_reverse, batch_y_len):
        # print(enc_inputs.size())
        # torch.Size([64, 1024, 128])
        ##############################################################################################

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # print(enc_outputs.size(), len(enc_self_attns), enc_self_attns[0].size())
        # torch.Size([64, 1024, 512]) 2 torch.Size([64, 12, 1214, 1214])

        dec_outputs, dec_self_attns, dec_enc_attns, \
        reverse_dec_outputs, reverse_dec_self_attns, reverse_dec_enc_attns = \
            self.decoder(dec_inputs, enc_outputs,
                using_reverse, batch_y_len, reverse_dec_inputs) 

        dec_outputs = self.projection(dec_outputs)
        gated_dec_outputs = F.sigmoid(self.gated_projection(dec_outputs))
        sep_dec_outputs = nn.ReLU()(self.projection2(dec_outputs))

        dec_logits = self.layerNorm(sep_dec_outputs * (1 - gated_dec_outputs) + dec_outputs * gated_dec_outputs)

        reverse_dec_outputs = self.projection(reverse_dec_outputs)
        gated_reverse_dec_outputs = F.sigmoid(self.gated_projection(reverse_dec_outputs))
        sep_reverse_dec_outputs = nn.ReLU()(self.projection2(reverse_dec_outputs))

        reverse_dec_logits = self.layerNorm(sep_reverse_dec_outputs * (1 - gated_reverse_dec_outputs) +
                                    reverse_dec_outputs * gated_reverse_dec_outputs)

        return dec_logits, reverse_dec_logits


