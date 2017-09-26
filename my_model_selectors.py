import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import traceback


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        #print("Constant selector");
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection based on BIC scores
        if self.verbose: print("="*10,"BIC","="*50);
        scores = {};
        for n in range(self.min_n_components,self.max_n_components+1):
            if self.verbose: print("=== n = %d" % n);
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, n))
                #BIC score
                #number of features is 4 for all, i.e,
                #grnd-ry, grnd-rx, grnd-ly, grnd-lx
                #norm-rx, norm-ry, norm-lx,norm-ly
                #delta-rx, delta-ry, delta-lx, delta-ly
                #norm-polar-rr, norm-polar-rtheta, norm-polar-lr, norm-polar-ltheta
                n_features = len(self.X[0]); #was hard coded to 4
                p = n**2 + 2*n*n_features - 1;
                s = -2 * hmm_model.score(self.X,self.lengths) + p*np.log(len(self.lengths));
                if self.verbose:
                    print("Size X,lengths is %.2f,%.2f" % (len(self.X),len(self.lengths)));
                    print("Score is: %f" % s);
                scores[s] = hmm_model;
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
        return scores[min(scores.keys())] if len(scores) > 0 else None;



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def model(self,word,n):
        try:
            X, lengths = self.hwords[word];
            hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            #logL_n_state_scores[word] = hmm_model.score(X,lengths);
            #n_model[word] = hmm_model;
            if self.verbose:
                #print("model created for {} with {} states".format(word, n), end=", ")
                #print("Size X,lengths is %.2f,%.2f" % (len(X),len(lengths)), end=", ");
                #print("Score is: %f" % logL_n_state_scores[word], end="; ");
                pass;
            return hmm_model.score(X,lengths), hmm_model;
        except ValueError:
            return None,None;
        except Exception as e:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n))
                print("Exception:",e);
                traceback.print_exc();
            return None,None;

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #implement model selection based on DIC scores
        if self.verbose: print("="*10,"DIC","="*50);
        scores = {};
        for n in range(self.min_n_components,self.max_n_components+1):
            if self.verbose: print("=== n-states = %d" % n);
            #get all logLs for this n state
            words = list(self.hwords.keys()); 
            words.pop(words.index(self.this_word));
            logL_n_state_scores = {};
            #n_model = {};

            #do first the query word
            logL = None;
            model = None;
            logL, model = self.model(self.this_word,n);
            if logL is not None and model is not None:
                logL_n_state_scores[self.this_word] = logL;
            else:
                if self.verbose: print("Failed to model %s for %d-states" % (self.this_word,n));
                continue;

            #calculate sum term
            for word in words:
                #do first the query word
                try:
                    X, lengths = self.hwords[word];
                    logL_n_state_scores[word] = model.score(X, lengths);
                except:
                    if self.verbose:
                        print("Failed to model %s for %d-states under %s" % (word,n,self.this_word));
            #calculate DIC
            try:
                sum_logL=sum([ logL_n_state_scores[word] for word in logL_n_state_scores if word != self.this_word]);
                dic = logL_n_state_scores[self.this_word] - sum_logL/(len(logL_n_state_scores)-1);
                scores[dic] = model;
                if self.verbose: print("DIC score: ",dic);
            except Exception as e:
                if self.verbose:
                    print("DIC failure on {} with {} states".format(self.this_word, n))
                    print("Exception:",e);
                    traceback.print_exc();

        return scores[max(scores.keys())] if len(scores) > 0 else None;

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection using CV

        #prepare training, testing pairs
        from sklearn.model_selection import KFold
        splits=3;
        if len(self.sequences) == 2: splits = 2;
        if len(self.sequences) < 2:
            return None;
        splitter = KFold(n_splits=splits);
        #if self.verbose:
            #print("default xlengths","------");
            #print(self.X)
            #print("------");
            #print(self.lengths);
            #print("------");
            #print("length: ",len(self.sequences));
            #print("sequences: ", self.sequences);
        train_test_indices = splitter.split(self.sequences); #[ ([train,..], [test,..]),([],[]),..]
        training = [];
        testing = [];
        for train, test in train_test_indices:
            if self.verbose: print("train: ", train)
            training.append(combine_sequences(train,self.sequences));
            if self.verbose: print("test: ", test)
            testing.append(combine_sequences(test,self.sequences));


        if self.verbose: print("="*10,"CV","="*50);
        scores = {};
        for n in range(self.min_n_components,self.max_n_components+1):
            if self.verbose: print("=== n = %d" % n);
            try:
                split_scores = [];
                for train, test in zip(training,testing):
                    hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train[0], train[1])
                    s = hmm_model.score(test[0],test[1]);
                    split_scores.append(s);
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, n))
                        print("Train size X,lengths is ",len(train[0])," ",len(train[1]));
                        print("Testing size X,lengths is ",len(test[0])," ",len(test[1]));
                        #print("training","------")
                        #print(len(train[0]),train[0]);
                        #print(len(train[1]),train[1]);
                        #print("testing","------")
                        #print(len(test[0]),test[0]);
                        #print(len(test[1]),test[1]);
                        print("Score is: %f" % s);
                scores[np.mean(split_scores)] = hmm_model;
                if self.verbose:
                    print("Avg score: ", np.mean(split_scores));
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
        return scores[max(scores.keys())] if len(scores) > 0 else None;
