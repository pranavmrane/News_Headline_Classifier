import pickle
import os


class NBClassifier:

    __slots__ = ['vocabulary', 'vocab_size', 'count_of_classes',
                 'doc_count_per_class', 'len_text_column',
                 'cond_prob_per_word', 'token_count_per_class']

    def __init__(self):
        """Load Values Required to Predict Class from File.
        """
        # Check if file exists
        print(os.path)
        if os.path.exists('Classifier_Code/classifier_details_nb.txt'):
            items = self.get_pickled_list('Classifier_Code/classifier_details_nb.txt')
        else:
            raise FileExistsError("File Not Found")

        # Save Values
        self.vocabulary = items[0]
        self.count_of_classes = items[1]
        self.doc_count_per_class = items[2]
        self.len_text_column = items[3]
        self.cond_prob_per_word = items[4]
        self.token_count_per_class = items[5]
        self.vocab_size = len(self.vocabulary)

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
        for word in doc.split(" "):
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
        score = [0.0] * self.count_of_classes
        for i in range(len(score)):
            score[i] = self.doc_count_per_class[i] * 1 / self.len_text_column

        tokens = self.validate_words(doc)
        if len(tokens) != 0:
            for i in range(self.count_of_classes):
                for token in tokens:
                    if token in self.cond_prob_per_word[i]:
                        score[i] *= self.cond_prob_per_word[i].get(token)
                    else:
                        score[i] *= (1 / (self.token_count_per_class[i]
                                          * self.vocab_size))
            max_score = score[0]
            class_label = 0
            for i in range(len(score)):
                if max_score < score[i]:
                    max_score = score[i]
                    class_label = i
            return self.get_true_class(class_label)
        else:
             return "Not Applicable"

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