import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os
import sys
import traceback


class ImportAndClean:

    __slots__ = ['dataFrame', 'stemmed', 'stopWordList']

    def __init__(self, file_address=None, columns_required = None):
        """Get dataset address and open file in Python

            Keyword arguments:
                file_address -- (string)
                columns_required -- the required columns from table --
                                    (list of numbers of columns to be retained)
        """
        # If path is not specified, then read from the same folder as .py file
        if file_address is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_separator = os.path.sep
            file_location = dir_path + dir_separator + "newsCorpora.csv"
        else:
            file_location = file_address
        required_columns = [1, 4]
        file_address = file_location
        columns_required = required_columns
        column_names = ["TITLE", "CATEGORY"]
        # Read .csv using pandas
        try:
            self.dataFrame = pd.read_csv(file_address, sep='\t',
                                    names=column_names,
                                    usecols=columns_required)
        except FileNotFoundError:
            print("File not found")

        except Exception:
            print("Exception Found")
            traceback.print_exc(file=sys.stdout)

        # self.dataFrame = self.dataFrame[0:10000]

        self.stemmed = PorterStemmer()
        self.stopWordList = stopwords.words('english')

    def get_text_column(self):
        """Clean Training and Testing Statements

            Return:
                stemmed_headlines -- Cleaned Headlines -- (list)
        """
        list_to_be_cleaned = []
        list_to_be_cleaned = self.dataFrame['TITLE']
        filtered_headlines = self.get_cleaned_list(list_to_be_cleaned)
        stemmed_headlines = self.get_stemmed_list(filtered_headlines)
        return stemmed_headlines

    def process_user_input(self, list_to_be_altered):
        """Clean User Inputs

            Keyword arguments:
                simple_input -- (list of headlines)

            Return:
                stemmed_headlines -- Cleaned Headlines -- (list)
        """
        filtered_headlines = self.get_cleaned_list(list_to_be_altered)
        stemmed_headlines = self.get_stemmed_list(filtered_headlines)
        return stemmed_headlines

    def get_cleaned_list(self, list_to_be_altered):
        """Remove Punctuations, Numbers

            Keyword arguments:
            list_to_be_altered -- (list of headlines)

            Return:
                filtered_sentences -- Cleaned Headlines -- (list)
        """
        filtered_sentences = []
        lower_clean = []
        # Convert to Lower Case and Remove Punctuation
        for sentence in list_to_be_altered:
            lower_clean.append(sentence.lower()\
                .replace('[{}]'.format(string.punctuation), ''))
        # Tokenize words in a Headline
        # Remove StopWords, Long Words, or Words than contain numbers
        for sentence in lower_clean:
            filtered_sentences.append(" ".join(word for word in sentence.split()
                if ((word not in self.stopWordList)
                    and (word.isalpha())
                    and (1 < len(word.lower()) < 15))))
        # print("Words in Headlines Cleaned")
        return filtered_sentences

    def get_stemmed_list(self, list_to_be_altered):
        """Stem Words

            Keyword arguments:
            list_to_be_altered -- (list of headlines)

            Return:
                filtered_sentences -- Cleaned Headlines -- (list)
        """
        stemmed_sentences = []
        for sentence in list_to_be_altered:
            stemmed_word = ""
            for word in sentence.split(' '):
                stemmed_word += " " + self.stemmed.stem(word)
            # Remove Additional Space Added at Beginning
            stemmed_word = stemmed_word[1:]
            stemmed_sentences.append(stemmed_word)
        # print("Stemming of Headlines Completed")
        return stemmed_sentences

    def get_label_column(self):
        """Convert characters to numbers for ease of use


            Return:
                category_numerical -- List of numbers -- (list)
        """
        category_numerical = self.dataFrame['CATEGORY']. \
            str.replace("b", "0"). \
            str.replace("t", "1"). \
            str.replace("e", "2"). \
            str.replace("m", "3")
        category_numerical = category_numerical.tolist()
        category_numerical = [int(i) for i in category_numerical]
        return category_numerical

    def read_headlines_from_file(self, file_location):
        """Read headlines from file, one headline per line

            Keyword arguments:
            file_location -- (string)

            Return:
                content -- List of headlines -- (list)
        """
        try:
            with open(file_location) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            return content
        except FileNotFoundError:
            print("File Not Found")

        except Exception:
            print("Exception Found")
            traceback.print_exc(file=sys.stdout)


    def make_headline_from_user_usable(self, headline):
        """When user gives single headline wrap in list for ease of use

            Keyword arguments:
            headline -- (string)

            Return:
                content -- List of headline -- (list)
        """
        list_of_headlines = []
        list_of_headlines.append(headline)
        return list_of_headlines

    def clean_for_prediction(self, input_for_prediction, multiple_files=None):
        """User input can be file location of headlines or individual headline

             Keyword arguments:
             input_for_prediction -- (string)
             multiple_files -- (boolean) - False for individual headline

             Return:
                 combined_lists -- List of 2 Lists -- (original headlines,
                                                    stemmed-cleaned headlines)
         """
        combined_lists = []
        if multiple_files is False:
            # print("Handling input for single Headline:")
            predictor_input = self. \
                make_headline_from_user_usable(input_for_prediction)
        else:
            #  print("Handling input for file containing Headlines:")
            predictor_input = self. \
                read_headlines_from_file(input_for_prediction)

        cleaned_list = self.process_user_input(predictor_input)

        combined_lists.append(predictor_input)
        combined_lists.append(cleaned_list)
        return combined_lists

if __name__ == '__main__':
    value = ImportAndClean()
