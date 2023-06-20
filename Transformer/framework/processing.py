import time, os
import torch, pickle
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
from sklearn import metrics

from framework.public_utilization import create_folder
import framework.config as config
from nltk.translate.bleu_score import sentence_bleu



def move_data_to_gpu(x, cuda, half=False):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")
    if cuda:
        x = x.cuda()
        if half:
            x = x.half()
    return x


def sequence_real_event(generator, pred_y):
    pred_y_only_event = [k for k in pred_y if k != generator.v2i[generator.start_string] and
                         k != generator.v2i[generator.end_string] and
                         k != generator.v2i[generator.pad_string]]
    return pred_y_only_event


def each_sample_acc(each_seq_y, pred_y):
    max_len = max(len(each_seq_y), len(pred_y))
    min_len = min(len(each_seq_y), len(pred_y))
    val_acc_sample = 0
    for j in range(min_len):
        if each_seq_y[j] == pred_y[j]:
            val_acc_sample += 1
    val_acc_sample /= max_len
    return val_acc_sample



def model_predict_transformer(generator, model, each_x, each_seq_y):
    device = each_x.device

    start_symbol_ind = generator.v2i[generator.start_string],
    end_symbol_ind = generator.v2i[generator.end_string]

    dec_input = torch.ones(1, 1).fill_(start_symbol_ind[0]).to(torch.long).to(device)

    model.eval()
    with torch.no_grad():
        enc_outputs, enc_self_attns = model.encoder(each_x)

        for i in range(generator.max_length - 1):

            dec_outputs, _, _ = model.decoder(dec_input, each_x, enc_outputs)

            projected = model.projection(dec_outputs[:, -1])

            _, next_word = torch.max(projected, dim=1)

            next_word = next_word.item()

            if next_word == end_symbol_ind: break

            dec_input = torch.cat([dec_input,
                                   torch.ones(1, 1).type_as(dec_input).fill_(next_word)], dim=1)

    pred_y = dec_input.cpu().detach().numpy()[0]
    pred_y_only_event = sequence_real_event(generator, pred_y)

    # print(pred_y_only_event)

    each_seq_y = each_seq_y
    each_seq_y_only_event = sequence_real_event(generator, each_seq_y)

    # print(each_seq_y_only_event)

    return pred_y_only_event, each_seq_y_only_event


def save_best_model(iteration, current_epoch, model, optimizer, models_dir, modelfile):
    save_out_dict = {'iteration': iteration,
                     'epoch': current_epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
    save_out_path = os.path.join(models_dir, modelfile + config.endswith)
    torch.save(save_out_dict, save_out_path)


def forward_greedy_search_validation(model, generate_func, cuda, generator, verbose=0):
    audio_names = []
    output_seq = []
    output_at = []
    acc_list = []
    bleu_list = []

    for num, data in enumerate(generate_func):
        if len(data) == 4:
            batch_x, batch_y_seq, batch_y_seq_len, batch_audio_names = data
        else:
            batch_x, batch_y_seq, batch_y_seq_len = data

        for j in range(len(batch_x)):
            each_x = np.expand_dims(batch_x[j], axis=0)
            each_x = move_data_to_gpu(each_x, cuda)
            each_seq_y = batch_y_seq[j]

            pred_y_only_event, each_seq_y_only_event = model_predict_transformer(generator, model,
                                                                     each_x, each_seq_y)

            sample_acc = each_sample_acc(pred_y_only_event, each_seq_y_only_event)

            reference = [each_seq_y_only_event]
            candidate = pred_y_only_event
            bleu = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

            output_at.append(list(set(pred_y_only_event)))
            output_seq.append(pred_y_only_event)
            audio_names.append(batch_audio_names[j])

            acc_list.append(sample_acc)
            bleu_list.append(bleu)

            if verbose:
                print('num: ', num, ' j:  ', j, sample_acc, bleu)

    sequential_acc = sum(acc_list) / len(acc_list)
    ave_bleu = sum(bleu_list) / len(bleu_list)
    if verbose:
        print('all acc: ', len(acc_list), 'average: ', sequential_acc)
        print('all bleu_list: ', len(bleu_list), 'average: ', ave_bleu)

    return sequential_acc, ave_bleu


def bibound2single(all):
    events = []
    for event in all:
        if event == 3 or event == 4:
            events.append('Alarm_bell_ringing')
        if event == 5 or event == 6:
            events.append('Blender')
        if event == 7 or event == 8:
            events.append('Cat')
        if event == 9 or event == 10:
            events.append('Dishes')
        if event == 11 or event == 12:
            events.append('Dog')
        if event == 13 or event == 14:
            events.append('Electric_shaver_toothbrush')
        if event == 15 or event == 16:
            events.append('Frying')
        if event == 17 or event == 18:
            events.append('Running_water')
        if event == 19 or event == 20:
            events.append('Speech')
        if event == 21 or event == 22:
            events.append('Vacuum_cleaner')
    return events


def testing_greedy_search(model, generate_func, cuda, generator, reverse, verbose=0, start_end_labels=False):
    output_seq = []
    output_at = []
    audio_names = []

    acc_list = []
    bleu_list = []
    for _, data in enumerate(generate_func):
        batch_x, batch_y_seq, batch_y_seq_len, batch_audio_names = data

        for j in range(len(batch_x)):
            each_x = np.expand_dims(batch_x[j], axis=0)
            each_x = move_data_to_gpu(each_x, cuda)
            each_seq_y = batch_y_seq[j]

            pred_y_only_event, each_seq_y_only_event = model_predict_transformer(generator, model, each_x, each_seq_y)

            if start_end_labels:
                pred_y_only_event = bibound2single(pred_y_only_event)
                each_seq_y_only_event = bibound2single(each_seq_y_only_event)

            sample_acc = each_sample_acc(pred_y_only_event, each_seq_y_only_event)

            reference = [each_seq_y_only_event]
            candidate = pred_y_only_event
            bleu = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

            audio_names.append(batch_audio_names[j])
            output_at.append(list(set(pred_y_only_event)))
            output_seq.append(pred_y_only_event)
            acc_list.append(sample_acc)
            bleu_list.append(bleu)

    ave_acc = sum(acc_list) / len(acc_list)
    ave_bleu = sum(bleu_list) / len(bleu_list)
    if verbose:
        print('all acc: ', len(acc_list), 'average: ', ave_acc)
        print('all bleu_list: ', len(bleu_list), 'average: ', ave_bleu)
    return audio_names, output_at, output_seq, acc_list, bleu_list, ave_acc, ave_bleu


def training_process(generator, model, models_dir, epochs, lr_init = 1e-3,
                                     learning_rate_decay=False,
                                     cuda=config.cuda):

    create_folder(models_dir)
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)

    train_val_data = pickle.load(open(config.training_val_test_file, 'rb'))
    train_audio_indexes = train_val_data['training_audio_ids']
    check_iter = int(len(train_audio_indexes) / config.batch_size)

    criterion = nn.CrossEntropyLoss()

    scheduler_decay_factor = 0.98
    decay_start_epoch = epochs/10 if epochs>999 else epochs/5
    scheduler_decay_patience = 5
    cooldown = 0
    clip_grad = 2.5
    if learning_rate_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=scheduler_decay_factor,
                                                               patience=scheduler_decay_patience,
                                                               verbose=False,
                                                               threshold=0.0001,
                                                               threshold_mode='rel',
                                                               cooldown=cooldown,
                                                               min_lr=0,
                                                               eps=1e-08)

    # Train on mini batches
    val_acc_list = []
    val_bleu_list = []

    each_iter_loss = []
    start_time = time.time()
    for iteration, all_data in enumerate(generator.generate_train()):
        train_bgn_time = time.time()
        batch_x, batch_y_sequence, batch_y_len = all_data
        sub_x = move_data_to_gpu(batch_x, cuda)
        sub_y = move_data_to_gpu(batch_y_sequence, cuda)

        model.train()

        optimizer.zero_grad()

        tgt_in = sub_y[:, :-1]
        tgt_y = sub_y[:, 1:]
        dec_logits = model(enc_inputs=sub_x, dec_inputs=tgt_in)

        loss = criterion(dec_logits.contiguous().view(-1, generator.words_num),
                         tgt_y.contiguous().view(-1))


        final_loss = loss

        final_loss.backward()

        optimizer.step()

        each_iter_loss.append(final_loss.item())

        current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]

        if config.show_training:
            print("iter: %s , E: %s / %s" % (iteration, '%.2f' % (iteration / check_iter), epochs),
                  "| loss: %.6f" % final_loss.item(),
                  "| l_p: %.6f" % loss.item(),
                  "| lr: ", current_lr,
                  "| Batch:", config.batch_size,
                  "| Time:", time.time() - start_time, )
            start_time = time.time()

        if iteration % check_iter == 0 and iteration > 0:
            epoch = iteration / check_iter

            val_sequential_acc, val_ave_bleu = \
            forward_greedy_search_validation(model=model,
                                                  generate_func=generator.generate_validate(data_type='validation'),
                                                  cuda=config.cuda, generator=generator)

            val_acc_list.append(val_sequential_acc)
            val_bleu_list.append(val_ave_bleu)

            print('epoch: ', epoch, ' val_bleu: %.6f' % val_ave_bleu, ' seq_acc: %.6f' % val_sequential_acc)

            # Reduce learning rate
            # check_itera_step = int(itera_step * accumulation_steps)
            # if lr_decay and (iteration % check_itera_step == 0 > 0):
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.9
            if learning_rate_decay and epoch > decay_start_epoch:
                scheduler.step(epoch)


        # Stop learning
        if iteration > epochs * check_iter:
            final_test = 1
            if final_test:
                val_sequential_acc, val_ave_bleu = \
                    forward_greedy_search_validation(model=model,
                                                     generate_func=generator.generate_validate(data_type='validation'),
                                                     cuda=config.cuda, generator=generator)
                val_acc_list.append(val_sequential_acc)
                val_bleu_list.append(val_ave_bleu)

                print('epoch: ', epoch, ' val_bleu: %.6f' % val_ave_bleu, ' seq_acc: %.6f' % val_sequential_acc)

            save_out_dict = {
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)

            print('Training is done!!!')
            break


def save_output(model_pred_event_seq_file, audio_names, acc_list, bleu_list, output_seq, generator,
                start_end_labels=False):
    with open(model_pred_event_seq_file, 'w') as f:
        for name, acc, bleu, event in zip(audio_names, acc_list, bleu_list, output_seq):
            f.write(name + '\t' + str(acc) + '\t' + str(bleu) + '\t\t\t')
            if len(event):
                length = len(event) - 1
                if length:  # 元素大于一
                    for i in range(length):
                        if start_end_labels:
                            f.write(event[i] + '\t')
                        else:
                            f.write(generator.i2v[event[i]] + '\t')
                if start_end_labels:
                    f.write(event[-1] + '\n')
                else:
                    f.write(generator.i2v[event[-1]] + '\n')
            else:
                f.write('\n')
            f.flush()




def testing_model(generator, model, model_dir, pred_event_seq_dir, pred_event_at_dir, reverse,
                          start_end_labels=False):
    create_folder(pred_event_seq_dir)
    create_folder(pred_event_at_dir)

    for num, file in enumerate(os.listdir(model_dir)):
        modelpath = os.path.join(model_dir, file)
        print('loading model: ', modelpath)
        if config.cuda:
            checkpoint = torch.load(modelpath)
        else:
            checkpoint = torch.load(modelpath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

        if config.cuda:
            model.cuda()

        generate_func = generator.generate_test()

        audio_names, output_at, output_seq, \
        acc_list, bleu_list, ave_acc, ave_bleu = testing_greedy_search(model=model,
                                                                       generate_func=generate_func,
                                                                       cuda=config.cuda, generator=generator, verbose=0,
                                                                       reverse=reverse,
                                                                       start_end_labels=start_end_labels)

        modelname = file.split(config.endswith)[0]
        model_pred_event_seq_file = os.path.join(pred_event_seq_dir,
                                                 'Tbleu_' + '%.5f' % ave_bleu
                                                 + '_seqacc_' + '%.5f'%ave_acc
                                                 + '_' + modelname +'.txt')
        save_output(model_pred_event_seq_file, audio_names, acc_list, bleu_list, output_seq, generator,
                    start_end_labels=start_end_labels)


        model_pred_event_at_file = os.path.join(pred_event_at_dir,
                                                'Tbleu_' + '%.5f' % ave_bleu
                                                + '_seqacc_' + '%.5f' % ave_acc
                                                + '_' + modelname + '.txt')
        save_output(model_pred_event_at_file, audio_names, acc_list, bleu_list, output_at, generator,
                    start_end_labels=start_end_labels)

        print('BLEU: ', ave_bleu)


def evaluate_audio_tagging(pred_event_at_dir):
    weak_label_set = config.test_data['weak_label_set']

    test_id = config.test_id
    tagging_truth_label_matrix = config.tagging_truth_label_matrix

    for file in os.listdir(pred_event_at_dir):
        filepath = os.path.join(pred_event_at_dir, file)
        id_list = []
        events_list = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                part = line.split('\t\t\t')
                audio_id = part[0].split('\t')[0]
                events = part[1].split('\n')[0].split('\t')
                id_list.append(audio_id)
                if len(events[0]):
                    events_list.append(events)
                else:
                    events_list.append([])

        pre_tagging_label_matrix = np.zeros(tagging_truth_label_matrix.shape)
        for num, each_id in enumerate(test_id):
            events = events_list[id_list.index(each_id)]
            if len(events):
                for event in events:
                    if event in weak_label_set:
                        pre_tagging_label_matrix[num, weak_label_set.index(event)] = 1

        auc_list = []
        for k in range(tagging_truth_label_matrix.shape[-1]):
            pre_tagging_label_matrix_each_event = pre_tagging_label_matrix[:, k]
            tagging_truth_label_matrix_each_event = tagging_truth_label_matrix[:, k]

            if np.sum(tagging_truth_label_matrix_each_event):
                test_auc = metrics.roc_auc_score(tagging_truth_label_matrix_each_event,
                                                 pre_tagging_label_matrix_each_event)
                auc_list.append(test_auc)
        ave_auc = np.mean(auc_list)


        tp = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix > 1.5)
        fn = np.sum(tagging_truth_label_matrix - pre_tagging_label_matrix > 0.5)
        fp = np.sum(pre_tagging_label_matrix - tagging_truth_label_matrix > 0.5)
        prec = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        fvalue = 2 * (prec * recall) / (prec + recall)

        print('F-score: ', fvalue, ' AUC:', ave_auc)

