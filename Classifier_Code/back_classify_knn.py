import pickle
import os
import math
import operator


class KNNClassifier:

    def __init__(self):
        """Load Values Required to Predict Class from File.
        """
        # Check if file exists
        if os.path.exists('Classifier_Code/classifier_details_knn.txt'):
            items = self.get_pickled_list('Classifier_Code/'
                                          'classifier_details_knn.txt')
        else:
            raise FileExistsError("File Not Found")

        # Save Values
        self.count_of_classes = items[0]
        self.number_of_training_docs = items[1]
        self.vocabulary = items[2]
        self.word_and_its_docs = items[3]
        self.numeric_column_train = items[4]

    def loadall(self, filename):
        """Read Values from Pickle

            Keyword arguments:
                filename

            Return:
                Values Stored in Generator Class
        """
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def get_pickled_list(self, filename):
        """Convert Values from Generator Class to List for Ease of Use

            Keyword arguments:
                filename

            Return:
                usable_items - list of variable values
        """
        items = self.loadall(filename)
        usable_items = []
        for i in items:
            usable_items.append(i)
        return usable_items

    def get_true_class(self, value):
        """Convert Numeric Class to Text for readability

            Keyword arguments:
                value -- (int value)

            Return:
                class -- (string)
        """
        if value == 0:
            return "Business"
        elif value == 1:
            return "Science and Technology"
        elif value == 2:
            return "Entertainment"
        else:
            return "Health"

    def validate_words(self, doc):
        """Returns words that are present in vocabulary

            Keyword arguments:
                doc -- (String)

            Return:
                verified_words -- (list of words)
        """
        verified_words = []
        for word in set(doc.split(" ")):
            if len(word) < 1 or word not in self.vocabulary:
                continue
            else:
                verified_words.append(word)

        return verified_words

    def simply_predict(self, doc):
        """Predict class of String

            Keyword arguments:
                doc -- input headline -- (String)

            Return:
                  self.get_true_class(class_label) -- class -- (string)
        """

        query_docids_with_values = {}

        query = self.validate_words(doc)

        if len(query) != 0:
            for i in range(len(query)):
                word_in_consideration = query[i]
                if word_in_consideration not in self.vocabulary or \
                                len(word_in_consideration) < 1:
                    continue
                else:
                    document_ids_for_word = \
                        self.word_and_its_docs[word_in_consideration]
                    word_frequency = len(document_ids_for_word)
                    qtfidf = (1 + math.log(1, 10)) \
                             * math.log(self.number_of_training_docs / word_frequency, 10)

                    for document_id in document_ids_for_word:
                        score \
                            = self.word_and_its_docs[word_in_consideration][
                                  document_id] * \
                              qtfidf

                        if document_id not in query_docids_with_values:
                            query_docids_with_values[document_id] = score
                        else:
                            score += query_docids_with_values[document_id]
                            query_docids_with_values[document_id] = score

            sorted_query_docids_with_values = sorted(
                query_docids_with_values.items(),
                key=operator.itemgetter(1), reverse=True)

            numeric_class = self.get_class_from(sorted_query_docids_with_values)

            return self.get_true_class(numeric_class)
        else:
            return "Not Applicable"

    def get_class_from(self, sorted_dictionary):
        k = 20
        class_instances_found = [0] * self.count_of_classes
        for pair in sorted_dictionary:
            # print(pair[0], pair[1])
            class_value = self.numeric_column_train[pair[0]]
            class_instances_found[class_value] += 1
            k = k - 1
            if k == 0:
                break;

        max_class = 0
        maximum_class_value = class_instances_found[max_class]
        for i in range(len(class_instances_found)):
            if class_instances_found[i] > maximum_class_value:
                max_class = i
                maximum_class_value = class_instances_found[max_class]

        return max_class

    def buffer_for_prediction(self, original_plus_modified_headlines):
        """This method is designed to collate all results together, 1 headline
                                                                or n headlines

            Keyword arguments:
                original_plus_modified_headlines -- List of 2 Lists --
                                (original headlines,stemmed-cleaned headlines)

            Return:
                  results - (String)
        """
        list_of_original_headlines = original_plus_modified_headlines[0]
        list_of_cleaned_headlines = original_plus_modified_headlines[1]
        results = ""
        for i in range(len(list_of_cleaned_headlines)):
            results += str(i+1) + ". " + list_of_original_headlines[i] + \
                       " -> " + self.simply_predict(list_of_cleaned_headlines[i]) + "\n"
        results = results[:-1]

        return results