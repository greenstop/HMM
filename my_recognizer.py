import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # implement the recognizer
    # return probabilities, guesses

    unknowns = test_set.get_all_Xlengths()
    probabilities = []
    guesses = []
    for unknown in unknowns:
        logL = {};
        X = unknowns[unknown][0];
        lengths = unknowns[unknown][1];
        for word in models:
            try:
                logL[word] = models[word].score(X,lengths);
            except:
                logL[word] = float("-inf");
        guesses.append(max((logL[k],k) for k in logL)[1])
        probabilities.append(logL.copy());
    return probabilities, guesses;

            
