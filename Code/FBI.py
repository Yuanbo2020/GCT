def model_predict_bidirection_fuse_decoder_reverse(generator, model, each_x,
                                                           each_reverse_x):
    alpha = 0.5

    device = each_x.device

    start_symbol_ind = generator.v2i[generator.start_string]
    start_symbol_ind_reverse = generator.v2i[generator.start_string_reverse]

    end_symbol_ind = generator.v2i[generator.end_string]

    dec_input = torch.ones(1, 1).fill_(start_symbol_ind).to(torch.long).to(device)
    reverse_dec_input = torch.ones(1, 1).fill_(start_symbol_ind_reverse).to(torch.long).to(device)

    with torch.no_grad():
        enc_outputs, enc_self_attns = model.encoder(each_x)

        reverse_enc_outputs, _ = model.encoder(each_reverse_x)

        for i in range(generator.max_length - 1):

            dec_outputs, _, _ = model.decoder(dec_inputs=dec_input,
                                              enc_outputs=enc_outputs)

            reverse_dec_outputs, _, _ = model.decoder_reverse(dec_inputs=reverse_dec_input,
                                                      enc_outputs=reverse_enc_outputs)

            ############################################################################################
            dec_outputs = dec_outputs[:, -1]
            dec_outputs = model.projection(dec_outputs)
            gated_dec_outputs = F.sigmoid(model.gated_projection(dec_outputs))
            sep_dec_outputs = nn.ReLU()(model.projection2(dec_outputs))
            dec_outputs = model.layerNorm(sep_dec_outputs * (1 - gated_dec_outputs) +
                                          dec_outputs * gated_dec_outputs)

            ############################################################################################
            reverse_dec_outputs = reverse_dec_outputs[:, -1]
            reverse_dec_outputs = model.projection(reverse_dec_outputs)
            gated_reverse_dec_outputs = F.sigmoid(model.gated_projection(reverse_dec_outputs))
            sep_reverse_dec_outputs = nn.ReLU()(model.projection2(reverse_dec_outputs))
            reverse_dec_outputs = model.layerNorm(sep_reverse_dec_outputs * (1 - gated_reverse_dec_outputs) +
                                                reverse_dec_outputs * gated_reverse_dec_outputs)

            contextual_projected = dec_outputs * alpha + reverse_dec_outputs * (1-alpha)
            ############################################################################################

            _, next_word = torch.max(contextual_projected, dim=1)

            next_word = next_word.item()

            if next_word == end_symbol_ind: break

            dec_input = torch.cat([dec_input,
                                   torch.ones(1, 1).type_as(dec_input).fill_(next_word)], dim=1)
            reverse_dec_input = torch.cat([reverse_dec_input,
                                   torch.ones(1, 1).type_as(reverse_dec_input).fill_(next_word)], dim=1)

    pred_y = dec_input.cpu().detach().numpy()[0]

    return pred_y


