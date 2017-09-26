
import numpy as np
import pandas as pd
from asl_data import AslDb

asl = AslDb() # initializes the database
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame

asl.df.ix[98,2] # look at the data available for an individual frame

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary

from asl_utils import test_features_tryit
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

test_features_tryit(asl)

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]

training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))

training.get_word_Xlengths('CHOCOLATE')

df_means = asl.df.groupby('speaker').mean()
df_means

asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df.head()

from asl_utils import test_std_tryit
df_std = asl.df.groupby('speaker').std()
df_std
test_std_tryit(df_std)

columns=['right-x', 'right-y', 'left-x', 'left-y']
title = lambda s: '\n'+'='*10+' '+s+' '+'='*(70-len(s))+'\n';

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
print(title("Column Names"), columns);

df_mu = asl.df.groupby('speaker').mean()
df_sigma = asl.df.groupby('speaker').std();
print(title("Means"),df_mu)
print(title("Std"),df_sigma)
for c in columns:
    asl.df[c+"-mu"]= asl.df['speaker'].map(df_mu[c])
    asl.df[c+"-sigma"]= asl.df['speaker'].map(df_sigma[c])

print(title("ZIP"),[t for t in zip(features_norm,columns)]);

for f,c in zip(features_norm,columns):
    asl.df[f]= (asl.df[c] - asl.df[c+"-mu"])/asl.df[c+"-sigma"]

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
asl.df["polar-rr"]= (asl.df["grnd-rx"]**2 + asl.df["grnd-ry"]**2)**(1/2)
asl.df["polar-lr"]= (asl.df["grnd-lx"]**2 + asl.df["grnd-ly"]**2)**(1/2)
asl.df["polar-rtheta"]= np.arctan2(asl.df["grnd-rx"], asl.df["grnd-ry"])
asl.df["polar-ltheta"]= np.arctan2(asl.df["grnd-lx"], asl.df["grnd-ly"])

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
asl.df["delta-rx"] = asl.df["right-x"].diff().fillna(0);
asl.df["delta-ry"] = asl.df["right-y"].diff().fillna(0);
asl.df["delta-lx"] = asl.df["left-x"].diff().fillna(0);
asl.df["delta-ly"] = asl.df["left-y"].diff().fillna(0);

features_custom = ['norm-polar-rr', 'norm-polar-rtheta', 'norm-polar-lr','norm-polar-ltheta']

asl.df["norm-polar-rr"]= (asl.df["norm-rx"]**2 + asl.df["norm-ry"]**2)**(1/2)
asl.df["norm-polar-lr"]= (asl.df["norm-lx"]**2 + asl.df["norm-ly"]**2)**(1/2)
asl.df["norm-polar-rtheta"]= np.arctan2(asl.df["norm-rx"], asl.df["norm-ry"])
asl.df["norm-polar-ltheta"]= np.arctan2(asl.df["norm-lx"], asl.df["norm-ly"])

import unittest

class TestFeatures(unittest.TestCase):

    def test_features_ground(self):
        sample = (asl.df.ix[98, 1][features_ground]).tolist()
        self.assertEqual(sample, [9, 113, -12, 119])

    def test_features_norm(self):
        sample = (asl.df.ix[98, 1][features_norm]).tolist()
        np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

    def test_features_polar(self):
        sample = (asl.df.ix[98,1][features_polar]).tolist()
        np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

    def test_features_delta(self):
        sample = (asl.df.ix[98, 0][features_delta]).tolist()
        self.assertEqual(sample, [0, 0, 0, 0])
        sample = (asl.df.ix[98, 18][features_delta]).tolist()
        self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))
                         
suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
unittest.TextTestRunner().run(suite)

import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()
    
show_model_stats(demoword, model)

print(title("custom"));
my_testword = 'CHOCOLATE'
model_custom, logL_custom = train_a_word(my_testword, 3, features_custom) # Experiment here with different parameters
show_model_stats(my_testword, model_custom)
print("logL = {}".format(logL_custom))

print(title("delta"));
my_testword = 'CHOCOLATE'
model_delta, logL_delta = train_a_word(my_testword, 3, features_delta) # Experiment here with different parameters
show_model_stats(my_testword, model_delta)
print("logL = {}".format(logL_delta))

print(title("norm"));
my_testword = 'CHOCOLATE'
model_norm, logL_norm = train_a_word(my_testword, 3, features_norm) # Experiment here with different parameters
show_model_stats(my_testword, model_norm)
print("logL = {}".format(logL_norm))

print(title("polar"));
my_testword = 'CHOCOLATE'
model_polar, logL_polar = train_a_word(my_testword, 3, features_polar) # Experiment here with different parameters
show_model_stats(my_testword, model_polar)
print("logL = {}".format(logL_polar))

print(title("ground"));
my_testword = 'CHOCOLATE'
model_grnd, logL_grnd = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
show_model_stats(my_testword, model_grnd)
print("logL = {}".format(logL_grnd))

get_ipython().magic('matplotlib inline')

import math
from matplotlib import (cm, pyplot as plt, mlab)

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()

print(title("custom"))
visualize(my_testword, model_custom)
print(title("norm"))
visualize(my_testword, model_norm)
print(title("delta"))
visualize(my_testword, model_delta)
print(title("polar"))
visualize(my_testword, model_polar)
print(title("ground"))
visualize(my_testword, model_grnd)

from my_model_selectors import SelectorConstant

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
word = 'VEGETABLE' # Experiment here with different words
model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()
print("Number of states trained in model for {} is {}".format(word, model.n_components))

from sklearn.model_selection import KFold

training = asl.build_training(features_ground) # Experiment here with different feature sets
word = 'VEGETABLE' # Experiment here with different words
word_sequences = training.get_word_sequences(word)
split_method = KFold()
print(len(word_sequences),word_sequences)
for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds

words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit
import imp

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()

print(title("Xlengths"))
word, lengths = Xlengths["FISH"]
for k in Xlengths:
    print(k,len(Xlengths[k][0]),Xlengths[k][1])

import my_model_selectors
imp.reload(my_model_selectors)
SelectorCV = my_model_selectors.SelectorCV

for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14, verbose=True).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))

import my_model_selectors
imp.reload(my_model_selectors)
SelectorBIC = my_model_selectors.SelectorBIC

for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14, verbose=True).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))

import my_model_selectors
imp.reload(my_model_selectors)
SelectorDIC = my_model_selectors.SelectorDIC

for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14, verbose=True).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))

import asl_test_model_selectors
imp.reload(asl_test_model_selectors)
TestSelectors = asl_test_model_selectors.TestSelectors

suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from my_model_selectors import SelectorConstant

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

models = train_all_words(features_ground, SelectorConstant)
print("Number of word models returned = {}".format(len(models)))

test_set = asl.build_test(features_ground)
print("Number of test set items: {}".format(test_set.num_items))
print("Number of test set sentences: {}".format(len(test_set.sentences_index)))

print(len(test_set.get_all_Xlengths().keys()))

from my_recognizer import recognize
from asl_utils import show_errors
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorCV

features = features_norm # change as needed
model_selector = SelectorBIC # change as needed

models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)

features = features_polar # change as needed
model_selector = SelectorDIC # change as needed

models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)

features = features_custom # change as needed
model_selector = SelectorCV # change as needed

models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)

from asl_test_recognizer import TestRecognize
suite = unittest.TestLoader().loadTestsFromModule(TestRecognize())
unittest.TextTestRunner().run(suite)

df_probs = pd.DataFrame(data=probabilities)
df_probs.head()

