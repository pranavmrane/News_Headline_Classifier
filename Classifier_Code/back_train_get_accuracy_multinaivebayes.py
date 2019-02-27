import pickle
import os
import pandas as pd


class NBModelBuilder:

    __slots__ = ['text_column', 'count_of_classes', 'cond_prob_per_word',
                 'classStrings', 'doc_count_per_class',
                 'token_count_per_class', 'vocabulary', 'text_column_train',
                 'numeric_column_train', 'text_column_test',
                 'numeric_column_test']

    def __init__(self, strings, labels, split_size):
        """initialize variables, split dataset for training and testing.

            Keyword arguments:
                strings -- (list of headlines)
                labels -- (list of values for headlines as numbers)
                split_size -- (test size representation, under 1)
        """
        self.text_column = strings
        self.count_of_classes = len(set(labels))
        self.cond_prob_per_word = {}
        self.classStrings = [""] * self.count_of_classes
        self.doc_count_per_class = [0] * self.count_of_classes
        self.token_count_per_class = [0] * self.count_of_classes
        self.vocabulary = set()
        self.text_column_train, self.numeric_column_train, \
        self.text_column_test, self.numeric_column_test \
            = self.split_dataset(strings, labels, split_size)

    def split_dataset(self, text_column, numeric_column, split_size):
        """Split Text and Numeric Lists

            Keyword arguments:
                text_column -- (list of headlines)
                numeric_column -- (list of values for headlines as numbers)
                split_size -- (test size representation, under 1)

            Return:
                 text_column_train, numeric_column_train, text_column_test,
               numeric_column_test -- Separated Lists -- (list)
        """
        total_rows = len(text_column)
        # Calculate rows for Test Dataset
        test_rows_count = int(float(total_rows * split_size))
        # Extract Rows to get Test Dataset
        text_column_test = text_column[-test_rows_count:]
        numeric_column_test = numeric_column[-test_rows_count:]
        # Delete Rows from main dataset to get Training Dataset
        text_column_copy = text_column
        del text_column_copy[-test_rows_count:]
        text_column_train = text_column_copy
        numeric_column_copy = numeric_column
        del numeric_column_copy[-test_rows_count:]
        numeric_column_train = numeric_column_copy

        return text_column_train, numeric_column_train, text_column_test, \
               numeric_column_test

    def validate_words(self, doc):
        """Returns words that are present in vocabulary

            Keyword arguments:
                doc -- (String)

            Return:
                verified_words -- (list of words)
        """
        verified_words = []
        for word in doc.split(" "):
            if len(word) < 1:
                continue
            else:
                verified_words.append(word)

        return verified_words

    def train(self):
        """Perform Training

            Return:
                 self.print_accuracy() - String Containing Accuracy and
                                            Confusion Matrix -- (String)
        """
        # print("Training Begins Now:")
        for i in range(self.count_of_classes):
            # Making dictionary within dictionary
            self.cond_prob_per_word[i] = {}

        for i in range(len(self.numeric_column_train)):
            # Comment Pending here
            self.doc_count_per_class[self.numeric_column_train[i]] = \
                self.doc_count_per_class[self.numeric_column_train[i]] + 1
            # Collect every word for every class in one single place
            self.classStrings[self.numeric_column_train[i]] += \
                self.text_column_train[i] + " "

        # Remove Additional Space Added from previous line
        for i in range(self.count_of_classes):
            self.classStrings[i] = self.classStrings[i][:-1]

        # Get Count of every word
        for i in range(self.count_of_classes):
            # Tokenize all words in a class
            tokens = self.classStrings[i].split(" ")
            # Record Count of words for every class
            self.token_count_per_class[i] = len(tokens)
            for token in tokens:
                # Recording every word, only 1 instance of a word will be
                # recorded
                if len(token) > 1:
                    self.vocabulary.add(token)
                    if token in self.cond_prob_per_word[i]:
                        # Word was discovered before, increment count of word
                        count = self.cond_prob_per_word[i].get(token)
                        self.cond_prob_per_word[i][token] = count + 1
                    else:
                        # If word is newly discovered, then add to dictionary
                        self.cond_prob_per_word[i][token] = 1

        # Save Conditional Probability Per word per class
        for i in range(self.count_of_classes):
            vocab_size = len(self.vocabulary)
            for key in self.cond_prob_per_word[i]:
                token = key
                value = self.cond_prob_per_word[i].get(token)
                # Calculating Conditional Probability of every word in class
                prob = (value + 1) / (self.token_count_per_class[i] 
                                      + vocab_size)
                # Save Conditional Probability
                self.cond_prob_per_word[i][token] = prob

        # Display value per word per class
        # for i in range(self.count_of_classes):
        #     if i == 0:
        #         for key in self.cond_prob_per_word[i]:
        #             token = key
        #             value = self.cond_prob_per_word[i].get(token)
        #             print(token, value)

        # Classifier will be saved every time we train
        self.save_classifier()
        return self.get_accuracy()

    def get_accuracy(self):
        """Calculate Accuracy

            Return:
                 self.print_accuracy() - String Containing Accuracy and
                                            Confusion Matrix -- (String)
        """
        result = []
        misclassify_count = 0
        # Get Predicted class for every headline
        for sentence in self.text_column_test:
            result.append(self.predict(sentence))

        # Get count of instances where predicted class doesn't match
        # with test dataset
        for i in range(len(result)):
            if self.numeric_column_test[i] != result[i]:
                misclassify_count = misclassify_count + 1

        # Calculate Accuracy
        accuracy = (len(self.text_column_test)
                    - misclassify_count)/len(self.text_column_test) * 100

        # Format Accuracy and call confusion matrix
        return ("The accuracy in % is:" + str("{:4.2f}".format(accuracy)) +
                "\n" + self.print_confusion_matrix(result))

    def print_confusion_matrix(self, predicted):
        """Get Confusion Matrix Using Pandas

            Return:
                 df_confusion - String Containing Confusion Matrix -- (String)
        """
        y_actual = pd.Series(self.numeric_column_test, name='Actual')
        y_predicted = pd.Series(predicted, name='Predicted')
        df_confusion = pd.crosstab(y_actual, y_predicted, rownames=['Actual'],
                                   colnames=['Predicted'], margins=True)
        return str(df_confusion)

    def save_classifier(self):
        """Save Variables that will allow us to calculate Accuracy
        """
        # Remove Existing File
        os.remove('Classifier_Code/classifier_details_nb.txt')
        # Ensure file is absent
        if not os.path.exists('Classifier_Code/classifier_details_nb.txt'):
            # Create file
            output = open('Classifier_Code/classifier_details_nb.txt', 'wb')
        # Open file and save variables
        pickle.dump(self.vocabulary, output)
        pickle.dump(self.count_of_classes, output)
        pickle.dump(self.doc_count_per_class, output)
        pickle.dump(len(self.text_column), output)
        pickle.dump(self.cond_prob_per_word, output)
        pickle.dump(self.token_count_per_class, output)

        output.close()

    def predict(self, doc):
        """Predict class of String

            Keyword arguments:
                doc -- input headline -- (String)

            Return:
                  class_label -- predicted class -- (int)
        """
        vocab_size = len(self.vocabulary)
        score = [0.0] * self.count_of_classes

        for i in range(len(score)):
            score[i] = self.doc_count_per_class[i] * 1 / len(self.text_column)
        tokens = doc.split(" ")

        for i in range(self.count_of_classes):
            for token in tokens:
                if token in self.cond_prob_per_word[i]:
                    score[i] *= self.cond_prob_per_word[i].get(token)
                else:
                    score[i] *= (1 / (self.token_count_per_class[i]
                                      * vocab_size))
        max_score = score[0]
        class_label = 0
        for i in range(len(score)):
            if max_score < score[i]:
                max_score = score[i]
                class_label = i
        # print("Score is", max_score)
        return class_label

if __name__ == '__main__':
    pass
