import copy
import torch

def FedAvg(w, args, c_global=None, res_caches=None):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
        for i in range(len(w)):
            tmp += w[i][k]
        tmp = torch.true_divide(tmp, len(w))
        w_avg[k].copy_(tmp)
    ############### SCAFFOLD ############################
    if args.fed == 'scaffold' and c_global is not None and res_caches is not None:
        if args.all_clients:
            client_num_per_round = args.num_users
        else:
            client_num_per_round = max(int(args.frac * args.num_users), 1)
        
        # update global model
        avg_weight = torch.tensor(
                        [
                            1 / client_num_per_round
                            for _ in range(client_num_per_round)
                        ],
                        device=args.device,
                    ) #by number of selected clients per round, not dependent on the local data size
        # y_pred = net_glob(inputs).cpu()
        # print(y_pred.detach().numpy())
        # print(f'avg_weight: {avg_weight.cpu().detach().numpy()}')
        c_delta_cache = list(zip(*res_caches))
        # print(f'c_delta_cache {c_delta_cache}')
        # update global control
        for c_g, c_del in zip(c_global, c_delta_cache):
            # print(f'before c_g {c_g.cpu().detach().numpy()}')
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1) #delta_c = sum of avg_weight * delta_c_i
            # print(f'c_del: {c_del.cpu().detach().numpy()}')
            c_g.data += (
                client_num_per_round / args.num_users
            ) * c_del #c_global = |S| / N * c_delta
            # print(f'c_g {c_g.cpu().detach().numpy()}')
        return w_avg, c_global
    return w_avg

