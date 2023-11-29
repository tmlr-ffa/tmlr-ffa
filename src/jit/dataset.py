from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from six.moves import range
from data import Data
import pickle

#
#==============================================================================

def pickle_load_file(filename):
    try:
        f =  open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    except Exception as e:
        print(e)
        print("Cannot load from file", filename)
        exit()

class Dataset(Data):
    """
        Class for representing dataset (transactions).
    """
    def __init__(self, filename=None, fpointer=None, mapfile=None,
            separator=',', use_categorical = False):
        super().__init__(filename, fpointer, mapfile, separator, use_categorical)

        # split data into X and y
        self.feature_names = self.names[:-1]
        self.nb_features = len(self.feature_names)
        self.use_categorical = use_categorical

        samples = np.asarray(self.samps)
        if not all(c.isnumeric() for c in samples[:, -1]):
            le = LabelEncoder()
            le.fit(samples[:, -1])
            samples[:, -1]= le.transform(samples[:, -1])
            self.class_names = le.classes_

        samples = np.asarray(samples, dtype=np.float32)
        self.X = samples[:, 0: self.nb_features]
        self.y = samples[:, self.nb_features]
        self.num_class = len(set(self.y))
        self.target_name = list(range(self.num_class))

        #print("c nof features: {0}".format(self.nb_features))
        #print("c nof classes: {0}".format(self.num_class))
        #print("c nof samples: {0}".format(len(self.samps)))

        # check if we have info about categorical features
        if (self.use_categorical):
            self.target_name = self.class_names

            self.binarizer = {}
            for i in self.categorical_features:
                self.binarizer.update({i: OneHotEncoder(categories='auto', sparse=False)})#,
                self.binarizer[i].fit(self.X[:,[i]])
        else:
            self.categorical_features = []
            self.categorical_names = []
            self.binarizer = []
        #feat map
        self.mapping_features()



    def train_test_split(self, test_size=0.2, seed=0):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=seed)


    def transform(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.binarizer != [])
            tx = []
            for i in range(self.nb_features):
                #self.binarizer[i].drop = None
                if (i in self.categorical_features):
                    self.binarizer[i].drop = None
                    tx_aux = self.binarizer[i].transform(x[:,[i]])
                    tx_aux = np.vstack(tx_aux)
                    tx.append(tx_aux)
                else:
                    tx.append(x[:,[i]])
            tx = np.hstack(tx)
            return tx
        else:
            return x

    def transform_inverse(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.binarizer != [])
            inverse_x = []
            for i, xi in enumerate(x):
                inverse_xi = np.zeros(self.nb_features)
                for f in range(self.nb_features):
                    if f in self.categorical_features:
                        nb_values = len(self.categorical_names[f])
                        v = xi[:nb_values]
                        v = np.expand_dims(v, axis=0)
                        iv = self.binarizer[f].inverse_transform(v)
                        inverse_xi[f] =iv
                        xi = xi[nb_values:]

                    else:
                        inverse_xi[f] = xi[0]
                        xi = xi[1:]
                inverse_x.append(inverse_xi)
            return inverse_x
        else:
            return x

    def transform_inverse_by_index(self, idx):
        if (idx in self.extended_feature_names):
            return self.extended_feature_names[idx]
        else:
            print("Warning there is no feature {} in the internal mapping".format(idx))
            return None

    def transform_by_value(self, feat_value_pair):
        if (feat_value_pair in self.extended_feature_names.values()):
            keys = (list(self.extended_feature_names.keys())[list( self.extended_feature_names.values()).index(feat_value_pair)])
            return keys
        else:
            print("Warning there is no value {} in the internal mapping".format(feat_value_pair))
            return None

    def mapping_features(self):
        self.extended_feature_names = {}
        self.extended_feature_names_as_array_strings = []
        counter = 0
        if (self.use_categorical):
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    for j, _ in enumerate(self.binarizer[i].categories_[0]):
                        self.extended_feature_names.update({counter:  (self.feature_names[i], j)})
                        self.extended_feature_names_as_array_strings.append("f{}_{}".format(i,j)) # str(self.feature_names[i]), j))
                        counter = counter + 1
                else:
                    self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                    self.extended_feature_names_as_array_strings.append("f{}".format(i)) #(self.feature_names[i])
                    counter = counter + 1
        else:
            for i in range(self.nb_features):
                self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                self.extended_feature_names_as_array_strings.append("f{}".format(i))#(self.feature_names[i])
                counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            if (i in self.categorical_features):
                readable_x.append(self.categorical_names[i][int(v)])
            else:
                readable_x.append(v)
        return np.asarray(readable_x)


    def test_encoding_transformes(self, X_train):
        # test encoding

        X = X_train[[0],:]

        print("Sample of length", len(X[0])," : ", X)
        enc_X = self.transform(X)
        print("Encoded sample of length", len(enc_X[0])," : ", enc_X)
        inv_X = self.transform_inverse(enc_X)
        print("Back to sample", inv_X)
        print("Readable sample", self.readable_sample(inv_X[0]))
        assert((inv_X == X).all())

        '''
        for i in range(len(self.extended_feature_names)):
            print(i, self.transform_inverse_by_index(i))
        for key, value in self.extended_feature_names.items():
            print(value, self.transform_by_value(value))
        '''
