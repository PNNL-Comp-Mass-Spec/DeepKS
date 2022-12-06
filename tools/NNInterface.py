from datetime import datetime
import json
import re
import torch
import torch.nn
import torch.utils.data
from prettytable import PrettyTable
from torchinfo_modified import summary
import sklearn.metrics
from matplotlib import pyplot as plt, rcParams
import pickle
import numpy as np
import sys
import pandas as pd
from numbers import Number
import collections
sys.path.append('../data/preprocessing/')
from PreprocessingSteps.get_kin_fam_grp import HELD_OUT_FAMILY

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 13
rcParams['font.serif'] = ['Palatino', 'Times', 'Times New Roman', 'Gentinum', 'URW Bookman', "Roman", "Nimbus Roman"]
class NNInterface:
    def __init__(self, model_to_train, loss_fn, optim = None, inp_size=None, inp_types=None, model_summary_name = None, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.model: torch.nn.Module = model_to_train
        self.criterion = loss_fn
        self.optimizer = optim
        self.device = device
        self.inp_size = inp_size
        self.inp_types = inp_types
        if optim is None and inp_size is None and inp_types is None and model_summary_name is None:
            return
        fp = open(model_summary_name, "w", encoding="utf-8")
        fp.write(str(self))
        fp.close()
    
    def __str__(self):
        try:
            self.representation = "\n" + "--- Model Summary ---\n" + str(ms := summary(self.model, device=self.device, input_size=self.inp_size, dtypes=self.inp_types, col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings = ["var_names"], verbose=0, col_width=50)) + "\n"
            self.model_summary = ms
            torch.cuda.empty_cache()
        except Exception as e:
            print("Failed to run model summary:", flush=True)
            print(e, flush=True)
            exit(1)
        return self.representation
    
    def train(self, train_loader, num_epochs=50, lr_decay_amount=1.0, lr_decay_freq=1, threshold = None, include_val = True, val_dl = None, verbose = 1, fold = 1, maxfold = 1, roc = False, savefile = False, cutoff = 0.5, metric = 'acc'):
        assert metric.lower().strip() in ['roc', 'acc'], "Scoring `metric` needs to be one of `roc` or `acc`."
        train_scores = []
        if verbose:
            print(f"--- Training ---", flush=True)
        lowest_loss = float('inf')
        epoch = 0
        if threshold is None:
            threshold = float("inf")
        while not ((lowest_loss < threshold and epoch >= num_epochs) or epoch >= 2*num_epochs):
            self.model.train()
            total_step = len(train_loader)
            if epoch % lr_decay_freq == 0 and epoch > 0:
                for param in self.optimizer.param_groups:
                    param['lr'] *= lr_decay_amount
            for b, (*X, labels) in enumerate(list(train_loader)):
                X = [x.to(self.device) for x in X]
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(*X)
                if outputs.size() == torch.Size([]):
                    outputs = outputs.reshape([1])
                torch.cuda.empty_cache()
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    loss = self.criterion(outputs, labels.long())
                elif isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    loss = self.criterion(outputs, labels.float())
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                print_every = max(len(train_loader) // 2, 1)
                if (b+1) % print_every == 0 and verbose:
                    if metric == "roc":
                        score = sklearn.metrics.roc_auc_score(labels.cpu(), outputs.data.cpu())

                    elif metric == "acc":
                        score = sklearn.metrics.accuracy_score(labels.cpu(), torch.heaviside(torch.sigmoid(outputs.data.cpu()).cpu() - cutoff, values=torch.tensor([0.])).cpu())

                    train_scores += [score]*len(labels)            
                    if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                        print ('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f}, Train {}: {:.2f}'
                            .format(epoch+1, num_epochs, b+1, total_step, loss.item(), metric, score))
                    elif isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                        print ('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f}, Train {}: {:.2f}'
                            .format(epoch+1, num_epochs, b+1, total_step, loss.item(), metric, score))
                
                lowest_loss = min(lowest_loss, loss.item())
            
            print(f"Overall Train {metric} for Epoch [{epoch}] was {sum(train_scores)/len(train_scores):.3f}")
            assert include_val, "Need to have `include_val == True` for K-fold cross-validation."
            if include_val:
                total_step = len(val_dl)
                if verbose:
                    accuracy, loss, _, _, _, _ = self.eval(val_dl, roc, savefile, cutoff, metric)
                    print ('VAL Epoch [{}/{}], Batch [{}/{}], Val Loss: {:.4f}, Val {}: {:.2f} <<<'
                        .format(epoch+1, num_epochs, b+1, total_step, loss, metric, accuracy))
                    

            epoch += 1
        return accuracy, loss
    
    def eval(self, dataloader, roc = False, savefile = False, cutoff = 0.5, metric = 'roc'):
        assert metric.lower().strip() in ['roc', 'acc'], "Scoring `metric` needs to be one of `roc` or `acc`."

        all_labels = []
        all_outputs = []
        all_preds = []
        avg_perf = []
        avg_loss = []
        self.model.eval()
        outputs = torch.Tensor([-1])
        with torch.no_grad():
            for *X, labels in list(dataloader):
                X = [x.to(self.device) for x in X]
                labels = labels.to(self.device)
                outputs = self.model(*X)
                torch.cuda.empty_cache()
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    labels = labels.long()
                elif isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    labels = labels.float()

                loss = self.criterion(outputs, labels)

                predictions = torch.Tensor([-1])
                performance = -1
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    predictions = torch.argmax(outputs.data.cpu(), dim=1).cpu()
                    

                elif isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    predictions = torch.heaviside(torch.sigmoid(outputs.data.cpu()).cpu() - cutoff, values=torch.tensor([0.]))
                    outputs = torch.sigmoid(outputs.data.cpu())

                if metric == 'acc':
                        performance = sklearn.metrics.accuracy_score(labels.cpu(), predictions)
                elif metric == 'roc':
                    scores = outputs.data.cpu()
                    performance = sklearn.metrics.roc_auc_score(labels.cpu(), scores)

                all_labels += labels.cpu().numpy().tolist()
                all_outputs += outputs.data.cpu().numpy().tolist()
                all_preds += predictions.cpu().numpy().tolist()
                avg_perf += [performance]*len(labels)
                avg_loss += [loss.item()]*len(labels)
                    
            return sum(avg_perf)/len(avg_perf), sum(avg_loss)/len(avg_loss), all_outputs, all_labels, all_preds, torch.sigmoid(torch.Tensor(all_outputs)).data.cpu().numpy().tolist()

    def get_all_conf_mats(self, tl, vl, tel, ho, savefile = "", cutoffs = [0.4, 0.45, 0.5, 0.55, 0.6]):
        set_labels = ['Train', 'Validation', 'Test', f'Held Out Family — {HELD_OUT_FAMILY}']
        for li, l in enumerate([tl, vl, tel, ho]):
            preds = []
            eval_res = self.eval(dataloader=l)
            outputs = [x if not isinstance(x, list) else x[0] for x in eval_res[2]]
            labels = eval_res[3]

            for cutoff in cutoffs:
                preds.append([1 if x > cutoff else 0 for x in outputs])

            fig, ax = plt.subplots(nrows = int(len(cutoffs)**1/2), ncols=int(np.ceil(len(cutoffs)/int(len(cutoffs)**1/2))), figsize = (12, 12))
            ax = np.asarray(ax)
            for i, fp in enumerate(preds):
                cm = sklearn.metrics.confusion_matrix(labels, fp, labels=['Decoy', 'Target'])
                sklearn.metrics.ConfusionMatrixDisplay(cm).plot(ax = ax.ravel()[i], im_kw = {'vmin' : 600, 'vmax' : 2000})
                ax.ravel()[i].set_title(f"Cutoff = {cutoffs[i]} | Acc = {(cm[0, 0] + cm[1, 1])/sum(cm.ravel()):3.3f}")
            
            if savefile:
                fig.savefig(savefile + "_" + set_labels[li] + ".pdf", bbox_inches='tight')

    def get_all_rocs_by_group(self, loader, kinase_order, savefile = "", kin_fam_grp_file = "../data/preprocessing/kin_to_fam_to_grp_817.csv"):
        kin_to_grp = pd.read_csv(kin_fam_grp_file).applymap(lambda c: re.sub(r"[\(\)\*]", "", c))
        kin_to_grp['Kinase'] = [f"{r['Kinase']}|{r['Uniprot']}" for _, r in kin_to_grp.iterrows()]
        kin_to_grp = kin_to_grp.set_index("Kinase").to_dict()['Group']
        fig = plt.figure(figsize=(12, 12))
        plt.plot([0, 1], [0, 1], color='r', linestyle='--', alpha = 0.5, linewidth=0.5, label = "Random Model")
        eval_res = self.eval(dataloader=loader)
        outputs: list[float] = eval_res[-1]
        labels: list[int] = eval_res[3]
        assert len(outputs) == len(labels) == len(kinase_order), f"Something is wrong in NNInterface.get_all_rocs_by_group; the length of outputs, the number of labels, and the number of kinases in `kinase_order` are not equal. (Respectively, {len(outputs)}, {len(labels)}, {len(kinase_order)}.)"
        grp_to_indices = collections.defaultdict(list[int])
        for i in range(len(outputs)):
            grp_to_indices[kin_to_grp[kinase_order[i]]].append(i)

        outputs_dd = collections.defaultdict(list[float])
        labels_dd = collections.defaultdict(list[int])
        for g, inds in grp_to_indices.items():
            outputs_dd[g] = [outputs[i] for i in inds]
            labels_dd[g] = [labels[i] for i in inds]

        for i, g in enumerate(grp_to_indices):
            self.roc_core(outputs_dd[g], labels_dd[g], i, line_labels=[g], linecolor=None)
        
        if savefile:
            fig.savefig(savefile + "_" + str(len(outputs_dd)) + ".pdf", bbox_inches='tight')

    def get_all_rocs(self, tl, vl, tel, ho, savefile = ""):
        fig = plt.figure(figsize=(12, 12))
        ii = 0
        for i, loader in enumerate([tl, vl, tel, ho]):
            eval_res = self.eval(dataloader=loader)
            outputs = eval_res[2]
            labels = eval_res[3]
            self.roc_core(outputs, labels, i, linecolor=None)
            ii = i
        if savefile:
            fig.savefig(savefile + "_" + str(ii + 1) + ".pdf", bbox_inches='tight')

    def roc_core(self, outputs, labels, i, linecolor = (0.5, 0.5, 0.5), line_labels = ['Train', 'Validation', 'Test', f'Held Out Family — {HELD_OUT_FAMILY}']):
        assert len(outputs) == len(labels), f"Something is wrong in NNInterface.roc_core; the length of outputs ({len(outputs)}) does not equal the number of labels ({len(labels)})"
        if linecolor is None:
            linecolor = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        
        roc_data = sklearn.metrics.roc_curve(labels, outputs)
        aucscore = sklearn.metrics.roc_auc_score(labels, outputs)
        sklearn.metrics.RocCurveDisplay(fpr=roc_data[0], tpr=roc_data[1]).plot(color=linecolor, linewidth=1, ax=plt.gca(), label=f"{line_labels[0 if len(line_labels) == 1 else i]} Set (AUC = {aucscore:.3f})")
        if aucscore > .98:
            self.inset_auc()

        plt.title("ROC Curves")
        plt.xticks([x/100 for x in range(0, 110, 10)])
        plt.yticks([x/100 for x in range(0, 110, 10)])
        plt.legend(loc='lower right')


    def inset_auc(self):
        ax = plt.gca()
        ax_inset = ax.inset_axes([0.5, 0.1, 0.45, 0.65])
        x1, x2, y1, y2 = 0, 0.2, 0.8, 1
        ax_inset.set_xlim(x1, x2)
        ax_inset.set_ylim(y1, y2)
        ax.indicate_inset_zoom(ax_inset, linestyle='dashed')
        ax_inset.set_xticks([x1, x2])
        ax_inset.set_yticks([y1, y2])
        ax_inset.set_xticklabels([x1, x2])
        ax_inset.set_yticklabels([y1, y2])
        ax_inset.plot(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), color='b', linewidth=1)

    def test(self, test_loader, verbose=True, roc=False, savefile=True, cutoff = 0.5, text = "Test Accuracy of the model", metric = "acc"):
        print("\n--- Testing ---", flush=True)
        

        if cutoff is None:
            cutoff = 0.5
            performance, _, outputs, labels, predictions, probabilities = self.eval(test_loader, roc, savefile, cutoff, metric)
            for i in range(5, 95 + 1, 5):
                cutoff = i/100
                print(f"Cutoff {cutoff} accuracy:", sklearn.metrics.accuracy_score(labels.cpu(), torch.heaviside(torch.sigmoid(outputs.data.cpu()).cpu() - cutoff, values=torch.tensor([0.]))))
        else:
            performance, _, outputs, labels, predictions, probabilities = self.eval(test_loader, roc, savefile, cutoff, metric)

        print('{}: {:.3f} %\n'.format(text, performance))


        if verbose == True:
            tab = PrettyTable(['Index', 'Label', 'Prediction'])
            num_rows = min(25, len(labels))
            tab.title = f'First {num_rows} labels mapped to predictions'
            for i in range(num_rows):
                tab.add_row([i, labels[i], predictions[i]])
            print(tab, flush=True)
        if verbose == 2:
            fp = open("../bin/probs.pkl", "wb")
            fl = open("../bin/labels.pkl", "wb")
            pickle.dump(probabilities.cpu().numpy(), fp)
            pickle.dump(labels.cpu().numpy(), fl)
        return predictions
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), open(path, "wb"))
    
    def save_eval_results(self, loader, path, kin_order = None):
        eval_res = self.eval(dataloader=loader)
        outputs = eval_res[-1]
        labels = eval_res[3]
        results_dict = {'labels': labels, 'outputs': outputs}
        if kin_order is not None:
            results_dict.update({'kin_order': kin_order})
        json.dump(results_dict, open(path, "w"), indent=4)
    
    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        
    @staticmethod
    def get_input_size(dl, leave_out_last=True):
        inp_sizes = []
        assert(isinstance(dl, torch.utils.data.DataLoader))
        dl = list(dl)
        iterab = dl[0] if not leave_out_last else dl[0][:-1]
        for X in iterab:
            assert(isinstance(X, torch.Tensor))
            inp_sizes.append(list(X.size()))
        return inp_sizes
    
    @staticmethod
    def get_input_types(dl, leave_out_last=True):
        types = []
        assert(isinstance(dl, torch.utils.data.DataLoader))
        dl = list(dl)
        iterab = dl[0] if not leave_out_last else dl[0][:-1]
        for X in iterab:
            assert(isinstance(X, torch.Tensor))
            types.append(X.dtype)
        return types
