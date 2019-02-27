import math
import operator
import os
import pickle
import pandas as pd


class KNNModelBuilder:

    def __init__(self, strings, labels, split_size):
        """initialize variables, split dataset for training and testing.

            Keyword arguments:
                strings -- (list of headlines)
                labels -- (list of values for headlines as numbers)
                split_size -- (test size representation, under 1)
        """
        # self.text_column = strings
        self.count_of_classes = len(set(labels))
        # self.cond_prob_per_word = {}
        # self.classStrings = [""] * self.count_of_classes
        # self.doc_count_per_class = [0] * self.count_of_classes
        # self.token_count_per_class = [0] * self.count_of_classes
        self.vocabulary = set()
        self.word_and_its_docs = {}
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

    def train(self):
        """Perform Training

            Return:
                 self.print_accuracy() - String Containing Accuracy and
                                            Confusion Matrix -- (String)
        """
        self.vocabulary, self.word_and_its_docs \
            = self.build_postings_list(self.text_column_train)
        self.calculateTfIdf()
        self.save_classifier()
        return self.get_accuracy()

    def build_postings_list(self, headlines):
        # print("Making postings list")
        completeVocabulary = set()
        postingList = {}

        for i in range(len(headlines)):
            for word in headlines[i].split(" "):
                if len(word) < 1:
                    abc = "Small word found, do nothing"
                # Word not present in dictionary
                elif word not in postingList:
                    completeVocabulary.add(word)
                    postingList[word] = dict()
                    postingList[word][i] = 1
                else:
                    # Word found in same document
                    if i in postingList[word]:
                        postingList[word][i] = postingList[word][i] + 1
                    # Word was already discovered before
                    else:
                        postingList[word][i] = 1

        return completeVocabulary, postingList

    def calculateTfIdf(self):
        # print("Get TfiDF")
        number_of_docs = len(self.text_column_train)
        docLength = [0.0] * number_of_docs

        for i in range(len(list(self.vocabulary))):
            word_in_consideration = list(self.vocabulary)[i]
            document_ids_for_word =\
                self.word_and_its_docs[word_in_consideration]
            word_frequency = len(document_ids_for_word)
            # if word_in_consideration == "dip":
            #     print(self.word_and_its_docs["dip"])
            #     print("Looking into issues")

            for document_id in document_ids_for_word:
                # print("Document id", document_id, "for word",
                # document_ids_for_word )
                word_count_for_document = document_ids_for_word[document_id]
                tf_idf = (1+math.log(word_count_for_document, 10)) \
                         * math.log(number_of_docs/word_frequency, 10)
                docLength[document_id] += math.pow(tf_idf, 2)
                self.word_and_its_docs[word_in_consideration][document_id] \
                    = tf_idf

        # print(self.word_and_its_docs["dow"])

        for i in range(number_of_docs):
            docLength[i] = math.sqrt(docLength[i])

        # Normalization Process
        for i in range(len(list(self.vocabulary))):
            word_in_consideration = list(self.vocabulary)[i]
            document_ids_for_word = \
                self.word_and_its_docs[word_in_consideration]

            for document_id in document_ids_for_word:
                self.word_and_its_docs[document_id] = \
                    self.word_and_its_docs[document_id]/docLength[document_id]

    def get_accuracy(self):
        """Calculate Accuracy

            Return:
                 self.print_accuracy() - String Containing Accuracy and
                                            Confusion Matrix -- (String)
        """
        # print("Commence Mass Training")
        result = []
        misclassify_count = 0
        iteration_counter = 0
        # print("Size of test dataset", len(self.numeric_column_test))
        # print("Size of test dataset", len(self.text_column_test))
        for test_headline in self.text_column_test:
            # iteration_counter += 1
            # if iteration_counter % 5000 == 0:
            #     print(iteration_counter, test_headline)
            result.append(self.get_query_values(test_headline))
        # print("Testing Predictions are here:")
        # print("Size of predicted dataset", len(result))
        for i in range(len(self.numeric_column_test)):
            if self.numeric_column_test[i] != result[i]:
                misclassify_count += 1

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

    def get_query_values(self, query):
        # print("Querying something")
        number_of_docs = len(self.text_column_train)
        query_docids_with_values = {}

        query = list(set(query.split(" ")))

        for i in range(len(query)):
            word_in_consideration = query[i]
            if word_in_consideration not in self.vocabulary or \
                    len(word_in_consideration) < 1:
                continue
            else:
                document_ids_for_word = \
                    self.word_and_its_docs[word_in_consideration]
                word_frequency = len(document_ids_for_word)
                qtfidf =  (1+math.log(1, 10)) \
                         * math.log(number_of_docs/word_frequency, 10)

                for document_id in document_ids_for_word:
                    score \
                        = self.word_and_its_docs[word_in_consideration][document_id] * \
                          qtfidf

                    if document_id not in query_docids_with_values:
                        query_docids_with_values[document_id] = score
                    else:
                        score += query_docids_with_values[document_id]
                        query_docids_with_values[document_id] = score

        # Sorting dictionary by its values to get highest confidence first
        # The type returned is list of tuples instead of dictionary
        sorted_query_docids_with_values \
            = sorted(query_docids_with_values.items(),
                     key=operator.itemgetter(1), reverse=True)
        # print(type(sorted_query_docids_with_values))
        # print(type(sorted_query_docids_with_values[1]))
        # print(type(sorted_query_docids_with_values[1][0]))
        # print(type(sorted_query_docids_with_values[1][1]))
        #sorted_query_docids_with_values = sorted_query_docids_with_values.reverse()

        return self.get_class_from(sorted_query_docids_with_values)

    def get_class_from(self, sorted_dictionary):
        #print("Get samll Class")
        k = 20
        class_instances_found = [0] * self.count_of_classes
        for pair in sorted_dictionary:
            # print(pair[0], pair[1])
            class_value = self.numeric_column_train[pair[0]]
            class_instances_found[class_value] += 1
            k = k - 1
            if k == 0:
                break

        max_class = 0
        maximum_class_value = class_instances_found[max_class]
        for i in range(len(class_instances_found)):
            if class_instances_found[i] > maximum_class_value:
                max_class = i
                maximum_class_value = class_instances_found[max_class]
        return max_class

    def save_classifier(self):
        """Save Variables that will allow us to calculate Accuracy
        """
        # print("Saving Classifier")
        # Remove Existing File
        os.remove('Classifier_Code/classifier_details_knn.txt')
        # Ensure file is absent
        if not os.path.exists('Classifier_Code/classifier_details_knn.txt'):
            # Create file
            output = open('Classifier_Code/classifier_details_knn.txt', 'wb')
        # Open file and save variables
        # output = open('classifier_details_knn.txt', 'wb')
        pickle.dump(self.count_of_classes, output)
        pickle.dump(len(self.text_column_train), output)
        pickle.dump(self.vocabulary, output)
        pickle.dump(self.word_and_its_docs, output)
        pickle.dump(self.numeric_column_train, output)
        output.close()




