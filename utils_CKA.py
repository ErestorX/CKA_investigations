def get_clean_CKA(json_summaries, model_t, model_t_name, model_c, model_c_name, data_loader, args):
    if model_c_name not in json_summaries.keys():
        if args.local_rank == 0:
            print('\t---Starting clean CKA computation with ' + model_c_name + '---')
        writer = SummaryWriter()
        modc_hooks = get_all_hooks(model_c, is_t2t='t2t' in model_c_name, is_performer=model_c_name.split('_')[3] == 'p')
        modt_hooks = get_all_hooks(model_t, is_t2t='t2t' in model_t_name, is_performer=model_t_name.split('_')[3] == 'p')
        metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks)

        with torch.no_grad():
            for it, (input, target) in enumerate(data_loader):
                do_log = (it % 10 == 0)
                _ = model_c(input)
                _ = model_t(input)
                update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", it, writer, do_log)
                for hook0 in modc_hooks:
                    for hook1 in modt_hooks:
                        hook0.clear()
                        hook1.clear()

        sim_mat = get_simmat_from_metrics(metrics_ct)
        json_summaries[model_c_name] = sim_mat.tolist()