# IMPORTS
# ------------------------------------------
import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis

import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------
IS_NOVEL = -1

class NovelNetwork(torch.nn.Module):
    # instance variables
    # ------------------
    model = None
    gmm = None
    device = None

    feat_layer = None
    known_labels = None
    threshold = None
    dist_metric = None
    confidence_delta = None

    # Nearest-Class Mean
    NearCM = None
    NearCM_delta = None
    
    def __init__(self, layers, known_labels, use_gpu=True):
        super().__init__()
        self.model = torch.nn.Sequential(layers)
        self.known_labels = torch.tensor(known_labels)
        
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    # FUNCTION: extract_feats( [], str ) => []
    # SUMMARY: provided a target layer and input, return feats from
    #          the target layer
    def extract_feats(self, input, target_layer):
        model = self._modules['model']
        cur_feats = input
        for cur_layer in model._modules:
            cur_feats = model._modules[cur_layer](cur_feats)
            if cur_layer == target_layer: return cur_feats
        
        raise ValueError("[Feature Extraction] Request layer not found!")

    def to_novel(self, input):
        input = input.to(dtype=int)
        bools = torch.tensor([x in self.known_labels for x in input])
        novel_input = torch.where(bools, input, -1)
        return novel_input

    def raw_predict(self, input):
        model = self._modules['model']
        return model(input).argmax(1)

    def predict(self, input):
        if self.threshold == None: 
            raise ValueError("Model is not yet trained!")

        raw_preds = np.array(self.raw_predict(input))
        feats = np.array(self.extract_feats(input, self.feat_layer).detach())
        gmm_preds = self.gmm.predict(feats)
        gmm_means = self.gmm.means_[gmm_preds]
        gmm_cov = self.gmm.covariances_

        std_dist = np.zeros(shape=feats.shape[0])
        for i, sample_cov in enumerate(gmm_cov):
            sample_inv = np.linalg.inv(sample_cov)
            sample_dist = mahalanobis(feats[i], gmm_means[i], sample_inv)
            std_dist[i] = sample_dist

        raw_preds[std_dist > self.threshold] = -1
        return raw_preds

    # FUNCTION: train( y, {} ) => void
    # SUMMARY: train the network, then fit a GMM to a sample of the 
    #          training data features
    def train(self, train_data, val_data, args, print_info=False):
        self.feat_layer = args['feat_layer']
        if 'dist_metric' in args:
            self.dist_metric = args['dist_metric']
        else: self.dist_metric = 'mahalanobis'

        print_every = args['print_every']
        feat_samp = args['feat_sample']
        min_clusters = args['min_g']
        max_clusters = args['max_g']
        epochs = args['epoch']
        lr = args['lr']

        # train neural network
        # ---------------------
        model = self._modules['model']
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = model.to(device=self.device)  # move the model parameters to CPU/GPU

        for _ in range(epochs):
            for t, (x, y) in enumerate(train_data):
                model.train()  # put model to training mode
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)

                scores = model(x)
                loss = F.cross_entropy(scores, y)

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()

                if t % print_every == 0 and print_info==True:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    self.check_accuracy(val_data, model)
                    print()
        
        # run coarse search over gaussian mixture model
        # ---------------------------------------------
        X_feats, _, y = self.GMM_batch(train_data, self.feat_layer, feat_samp)

        best_aic = float('inf')
        best_gmm = None
        for n_comp in range(min_clusters, max_clusters):
            cur_gmm = GaussianMixture(n_components=n_comp)
            cur_gmm.fit(X_feats, y)
            cur_aic = cur_gmm.aic(X_feats)
            if cur_aic < best_aic:
                best_aic = cur_aic
                best_gmm = cur_gmm

        self.gmm = best_gmm
        print("Best # components: ", best_gmm.n_components)
        
        # Set the mahalanobis threshold 
        # ------------------------------
        X_test_feats, X_test, y_test = self.GMM_batch(val_data, self.feat_layer, feat_samp)

        # Set Nearest-Class Means
        # ------------------------------
        self.NearCM = np.zeros((len(self.known_labels), X_test_feats.shape[1]))
        for i in self.known_labels:
            self.NearCM[i] = np.mean(X_test_feats[y_test == i], axis=0)

        best_acc = self.set_threshold(X_test, X_test_feats, y_test)
        return best_acc
    
    def GMM_batch(self, loader, target_layer, num_batches=50):

        X_feats = np.array([])
        X = torch.tensor([])
        y = torch.tensor([], dtype=torch.float32)
        for i in range(0, num_batches):
            cur_batch = iter(loader)
            X_batch, y_batch = next(cur_batch)

            X_batch = X_batch.to(self.device)
            cur_feats = self.extract_feats(X_batch, target_layer)

            X = torch.cat((X.cpu(), X_batch.cpu()), axis=0)
            y = torch.cat((y.cpu(), y_batch.cpu()), axis=0)

            if i == 0: X_feats = X_feats.reshape((0, cur_feats.shape[1]))
            X_feats = np.concatenate((X_feats, cur_feats.detach().cpu()), axis=0)

        return X_feats, X, y


    def check_accuracy(self, loader, get_wrong=False):
        model = self._modules['model']
        device = self.device

        if loader.dataset.train: print('Checking accuracy on validation set')
        else: print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        wrong_imgs = []
        wrong_labels = []
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                scores = model(x)
                _, preds = scores.max(1)

                img_wr = x[preds != y].to('cpu')
                label_wr = preds[preds != y].to('cpu')
                wrong_imgs.extend(img_wr)
                wrong_labels.extend(label_wr)

                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
        if get_wrong == True: 
            return wrong_imgs, np.array(wrong_labels)
    
    def set_threshold(self, X, X_test_feats, y_test, print_info=False):
        model = self._modules['model']
        dist_metric = self.dist_metric
        model_preds = self.to_novel(self.raw_predict(X))
        y_test = np.array(self.to_novel(y_test))

        gmm_preds = self.gmm.predict(X_test_feats)
        gmm_means = self.gmm.means_[gmm_preds]
        gmm_cov = self.gmm.covariances_

        # Calculate distance between instance and its clsoest center
        # ----------------------------------------------
        std_dist = np.zeros(shape=X_test_feats.shape[0])
        for i, sample in enumerate(X_test_feats):
            cur_sample = sample.reshape(-1, 1)
            cur_mean = gmm_means[i].reshape(gmm_means[i].shape[0], 1)
            
            iv = np.linalg.inv(gmm_cov[gmm_preds[i]])

            if dist_metric == 'euclidean': sample_dist = np.sum(np.abs(cur_sample - cur_mean))
            elif dist_metric == 'mahalanobis': sample_dist = mahalanobis(cur_sample, cur_mean, iv)
            else: raise ValueError('unsupported distance metric')
            std_dist[i] = sample_dist
        # ----------------------------------------------

        # find the best novelty threshold
        min_thresh = min(std_dist)
        max_thresh = max(std_dist)
        threshold = min_thresh
        thresh_delta = abs(max_thresh - min_thresh)/1000
        
        cur_it = 0
        best_acc = - float('inf')
        best_threshold = None
        while threshold < max_thresh:
            cur_preds = np.array(model_preds.clone())
            cur_preds[std_dist > threshold] = IS_NOVEL

            cur_acc = np.sum(cur_preds == y_test) / len(y_test)
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_threshold = threshold

            if print_info==True:
                plt.figure(cur_it)
                for i, label in enumerate(['novel', 'known']):
                    cur = i - 1
                    plt.scatter(X_test_feats[cur_preds == cur, 0], X_test_feats[cur_preds == cur, 1], label=label)
                            
                    plt.scatter(gmm_means[:, 0], gmm_means[:, 1], marker='x', color='red')
                    #plt.title('Delta = {DELTA}, Acc = {ACC}'.format(DELTA=best_threshold, ACC=best_acc))
                plt.legend()
                plt.axis('off')
                plt.savefig('train-plot.jpeg')
                plt.show()

            threshold += thresh_delta
            cur_it = cur_it + 1

        # plot best thresholding
        cur_preds = np.array(model_preds.clone())
        cur_preds[std_dist > best_threshold] = IS_NOVEL
        cur_preds[std_dist <= best_threshold] = 0

        if print_info==True:
            for i, label in enumerate(['novel', 'known']):
                cur = i - 1
                plt.figure(cur_it)
                plt.scatter(X_test_feats[cur_preds == cur, 0], X_test_feats[cur_preds == cur, 1], label=label)
                        
                plt.scatter(gmm_means[:, 0], gmm_means[:, 1], marker='x', color='red')
                #plt.title('Delta = {DELTA}, Acc = {ACC}'.format(DELTA=best_threshold, ACC=best_acc))
                plt.legend()
                plt.axis('off')
            plt.savefig('train-plot.jpeg')
            plt.show()
        
        # compute model-confidence score
        # ------------------------------
        best_delta = None
        best_acc = -1
        for delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            conf_probs = np.array(model(X).detach())
            conf_preds = np.argmax(conf_probs, axis=1)
            conf_max = np.max(conf_probs, axis=1)
            conf_preds[np.where((conf_max > 1/len(self.known_labels) - delta) & (conf_max < 1/len(self.known_labels) + delta))] = IS_NOVEL

            cur_acc = np.sum(conf_preds == y_test) / len(y_test)
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_delta = delta

        self.confidence_delta = best_delta
        # ------------------------------

        # compute Nearest-Class Mean threshold
        # ---------------------------------
        all_NCM_dist = np.zeros((X_test_feats.shape[0]))#np.abs( np.array(X_test_feats) - np.array(self.NearCM) )

        for i, sample in enumerate(X_test_feats):
            cur_sample = sample.reshape(-1, 1)
            sample_dist = np.abs(cur_sample - np.array(self.NearCM))
            all_NCM_dist[i] = np.min(np.sum(sample_dist, axis=1))

        max_dist, min_dist = np.max(all_NCM_dist), np.min(all_NCM_dist)

        best_NCM = None
        best_NCM_acc = -1
        cur_delta = min_dist
        while cur_delta < max_dist:
            NCM_preds = np.array(model_preds.clone())
            NCM_preds[all_NCM_dist > cur_delta] = -1

            NCM_acc = np.sum(NCM_preds == y_test) / len(y_test)
            if NCM_acc > best_NCM_acc:
                best_NCM_acc = NCM_acc
                best_NCM = cur_delta
            
            cur_delta += (max_dist - min_dist)/(1000)

        self.NearCM_delta = best_NCM
        # ---------------------------------

        self.threshold = best_threshold
        print('------------------------------')
        print('Distance: ', self.dist_metric)
        print("Baseline Acc: ", np.sum(np.array(model_preds) == y_test) / len(y_test))
        print("Validation Acc: ", best_acc)
        print("Best Threshold: ", best_threshold)
        print('------------------------------')
        
        return best_acc

    def test_analysis(self, loader, get_wrong=False, print_info=False):
        model = self._modules['model']
        dist_metric = self.dist_metric
        X_test_feats, X_test, y_test = self.GMM_batch(loader, self.feat_layer, 20)
        y_test = np.array(self.to_novel(y_test))
        y_test = np.array(y_test)

        preds = torch.tensor(self.predict(X_test))
        preds = np.array(self.to_novel(preds))

        gmm_preds = self.gmm.predict(X_test_feats)
        gmm_means = self.gmm.means_[gmm_preds]
        gmm_cov = self.gmm.covariances_[gmm_preds]
        std_dist = np.zeros(shape=X_test_feats.shape[0])
        for i, sample in enumerate(X_test_feats):
            cur_sample = sample.reshape(-1, 1)
            cur_mean = gmm_means[i].reshape(gmm_means[i].shape[0], 1)
            
            iv = np.linalg.inv(gmm_cov[gmm_preds[i]])
            if dist_metric == 'euclidean': sample_dist = np.sum(np.abs(cur_sample - cur_mean))
            elif dist_metric == 'mahalanobis': sample_dist = mahalanobis(cur_sample, cur_mean, iv)
            else: raise ValueError('unsupported distance metric')
            
            std_dist[i] = sample_dist
        
        preds[std_dist > self.threshold] = IS_NOVEL

        acc = np.sum(preds == y_test) / len(y_test)
        preds[preds > IS_NOVEL] = 0

        if print_info==True:
            for i, label in enumerate(['novel', 'known']):
                    cur = i - 1
                    plt.figure('Test Analysis')
                    plt.scatter(X_test_feats[preds == cur, 0], X_test_feats[preds == cur, 1], label=label)
                            
                    plt.scatter(self.gmm.means_[:, 0], self.gmm.means_[:, 1], marker='x', color='red')
                    plt.title('Delta = {DELTA}, Acc = {ACC}'.format(DELTA=self.threshold, ACC=acc))
                    plt.legend()
            plt.savefig('test-plot.jpeg')
            plt.show()

            # Ground truth plot
            plt.figure('Ground Truth')
            plt.scatter(X_test_feats[y_test == IS_NOVEL, 0], X_test_feats[y_test == IS_NOVEL, 1], label='novel')
            plt.scatter(X_test_feats[np.where((y_test == 0) | (y_test == 1)), 0], X_test_feats[np.where((y_test == 0) | (y_test == 1)), 1], label='known')

            plt.scatter(self.gmm.means_[:, 0], self.gmm.means_[:, 1], marker='x', color='red')
            plt.title('Delta = {DELTA}, Acc = {ACC}'.format(DELTA=self.threshold, ACC=acc))
            plt.legend()
            plt.savefig('test-truth-plot.jpeg')
            plt.show()

        info = {}

        raw_preds = np.array(self.raw_predict(X_test))
        raw_acc = np.sum(raw_preds == y_test) / len(y_test)
        true_novel = y_test[y_test == -1]

        # compute model-confidence score
        # ------------------------------
        conf_probs = np.array(model(X_test).detach())
        conf_preds = np.argmax(conf_probs, axis=1)
        conf_max = np.max(conf_probs, axis=1)
        conf_preds[np.where((conf_max > 1/len(self.known_labels) - self.confidence_delta) & (conf_max < 1/len(self.known_labels) + self.confidence_delta))] = IS_NOVEL

        cur_acc = np.sum(conf_preds == y_test) / len(y_test)
        
        info['conf'] = {'delta' : self.confidence_delta, 
                        'acc' : cur_acc,
                        'novel_recall' : np.sum(conf_preds[y_test == -1] == true_novel) / len(true_novel)}


        # compute Nearest-Class Mean
        # ----------------------------
        all_NCM_dist = np.zeros((X_test_feats.shape[0]))
        for i, sample in enumerate(X_test_feats):
            cur_sample = sample.reshape(-1, 1)
            sample_dist = np.abs(cur_sample - np.array(self.NearCM))
            all_NCM_dist[i] = np.min(np.sum(sample_dist, axis=1))

        NCM_preds = np.array(raw_preds.copy())
        NCM_preds[all_NCM_dist > self.NearCM_delta] = -1
        info['NCM'] = {'acc' : np.sum(NCM_preds == y_test) / len(y_test),
                    'recall': np.sum(NCM_preds[y_test == -1] == true_novel) / len(true_novel)
                        }
        

        # compute recall on novel class
        # ------------------------------
        pred_novel = preds[y_test == -1]

        info['novel_recall'] =  np.sum(pred_novel == true_novel) / len(true_novel)
        # ------------------------------

        return acc, raw_acc, info

    def plot_feats(self, data, target_layer):
        X_feats, X, y = self.GMM_batch(data, target_layer, 20)
        y = self.to_novel(y)
        y[y != IS_NOVEL] = 0 

        plt.figure()
        plt.scatter(X_feats[y == IS_NOVEL, 0], X_feats[y == IS_NOVEL, 1], label='novel')
        plt.scatter(X_feats[y == 0, 0], X_feats[y == 0, 1], label='known')

        plt.axis('off')
        plt.savefig('feats-{layer}.jpeg'.format(layer=target_layer))
