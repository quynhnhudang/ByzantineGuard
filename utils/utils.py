import os
import torch
from datetime import datetime
import pickle

def save_model(start_time, args, net_glob, idxs_users):
    try:
        now = datetime.now()
        print("Total time for the training: {} seconds ---".format(now - start_time))
        now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

        file = f'{args.fed}_{args.adv_eps}_{args.dataset}_{args.model}_{args.local_ep}_nParties_{len(idxs_users)}_{args.sampling}__{args.classwise}_{args.alpha}_model'
        model_name = '{}_{}.pt'.format(file, now)
        filepath = args.filepath
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        f_model = os.path.join(filepath, model_name)
        torch.save(net_glob.state_dict(), f_model)
        
        print('The final model saved to: ', f_model)
        
        return f_model
    except Exception as e:
        print(f'Error saving model: {e}')

def save_model_performance(args, rounds_test_accuracy,
                           rounds_train_loss, 
                           rounds_adv_test_accuracy, 
                           rounds_adv_test_accuracy_2,
                           idxs_users):
    try:
        output = {}
        output['rounds_test_accuracy'] = rounds_test_accuracy
        output['rounds_train_loss'] = rounds_train_loss
        output['rounds_adv_test_accuracy'] = rounds_adv_test_accuracy
        output['rounds_adv_test_accuracy_2'] = rounds_adv_test_accuracy_2

        temp=args.filepath
        filename = f'{args.fed}_{args.adv_eps}_{args.dataset}_{args.model}_{args.local_ep}_nParties_{len(idxs_users)}_{args.sampling}_{args.classwise}_{args.alpha}_output.out'
        filepath = os.path.join(temp, filename)
        print('filepath ', filepath)
        outfile = open(filepath,'wb')
        pickle.dump(output, outfile)
    except Exception as e:
            print(f'Error saving model performance: {e}')