from datetime import datetime
import torch
from prettytable import PrettyTable
from torchinfo_modified import summary
import sklearn.metrics
from matplotlib import pyplot as plt, rcParams
import pickle
import sys
sys.path.append('../data/preprocessing/')
from PreprocessingSteps.get_kin_fam_grp import HELD_OUT_FAMILY

rcParams['font.family'] = 'Palatino'
rcParams['font.size'] = 13
class NNInterface:
    def __init__(self, model_to_train, loss_fn, optim, inp_size=(100, 15), inp_types=[torch.long], model_summary_name = "model_summary.txt", device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.model = model_to_train
        self.criterion = loss_fn
        self.optimizer = optim
        self.device = device
        self.inp_size = inp_size
        self.inp_types = inp_types
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
    
    def train(self, train_loader, num_epochs=50, lr_decay_amount=1.0, lr_decay_freq=1, thsh = None, include_val = True, val_dl = None, verbose = 1, fold = 1, maxfold = 1, roc = False, savefile = False, cutoff = 0.5, metric = 'acc'):
        assert metric.lower().strip() in ['roc', 'acc'], "Scoring `metric` needs to be one of `roc` or `acc`."
        train_scores = []
        if verbose:
            print(f"--- Training ---", flush=True)
        lowest_loss = float('inf')
        epoch = 0
        if thsh is None:
            thsh = float("inf")
        while not ((lowest_loss < thsh and epoch >= num_epochs) or epoch >= 2*num_epochs):
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

                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    predictions = torch.argmax(outputs.data.cpu(), dim=1).cpu()
                    

                elif isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    predictions = torch.heaviside(torch.sigmoid(outputs.data.cpu()).cpu() - cutoff, values=torch.tensor([0.]))

                if metric == 'acc':
                        performance = sklearn.metrics.accuracy_score(labels.cpu(), predictions)
                elif metric == 'roc':
                    scores = outputs.data.cpu()
                    performance = sklearn.metrics.roc_auc_score(labels.cpu(), scores)
                
                if roc:
                    self.plot_roc(labels, outputs, savefile)

                all_labels += labels.cpu().numpy().tolist()
                all_outputs += outputs.data.cpu().numpy().tolist()
                all_preds += predictions.cpu().numpy().tolist()
                avg_perf += [performance]*len(labels)
                avg_loss += [loss.item()]*len(labels)
                    
            return sum(avg_perf)/len(avg_perf), sum(avg_loss)/len(avg_loss), all_outputs, all_labels, all_preds, torch.sigmoid(outputs.data.cpu()).cpu()

    def get_all_rocs(self, tl, vl, tel, ho, savefile = ""):
        fig = plt.figure(figsize=(12, 12))
        linecolors = ['orange', 'violet', (0, .5, 0), 'blue']
        set_labels = ['Train', 'Validation', 'Test', f'Held Out Family — {HELD_OUT_FAMILY}']
        for i, loader in enumerate([tl, vl, tel, ho]):
            eval_res = self.eval(dataloader=loader)
            outputs = eval_res[2]
            labels = eval_res[3]
            roc_data = sklearn.metrics.roc_curve(labels, outputs)
            aucscore = sklearn.metrics.roc_auc_score(labels, outputs)
            sklearn.metrics.RocCurveDisplay(fpr=roc_data[0], tpr=roc_data[1]).plot(color=linecolors[i], linewidth=1, ax=plt.gca(), label=f"{set_labels[i]} Set (AUC = {aucscore:.3f})")
            if aucscore > .98:
                self.inset_auc()

        plt.title("ROC Curves")
        plt.plot([0, 1], [0, 1], color='r', linestyle='--', alpha = 0.5, linewidth=0.5, label = "Random Model")
        plt.xticks([x/100 for x in range(0, 110, 10)])
        plt.yticks([x/100 for x in range(0, 110, 10)])
        plt.legend(loc='lower right')
        if savefile:
            fig.savefig(savefile + "_" + str(len(linecolors)) + ".pdf", bbox_inches='tight')

    def inset_auc(self):
        ax = plt.gca()
        axins = ax.inset_axes([0.5, 0.1, 0.45, 0.65])
        x1, x2, y1, y2 = 0, 0.2, 0.8, 1
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        ax.indicate_inset_zoom(axins, linestyle='dashed')
        axins.set_xticks([x1, x2])
        axins.set_yticks([y1, y2])
        axins.set_xticklabels([x1, x2])
        axins.set_yticklabels([y1, y2])
        axins.plot(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), color='b', linewidth=1)

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

        if roc:
            self.plot_roc(labels, outputs, savefile)

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
    
    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), open(path, "wb"))
    
    @staticmethod
    def load_model(path):
        return pickle.load(open(path, "rb"))
        
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
