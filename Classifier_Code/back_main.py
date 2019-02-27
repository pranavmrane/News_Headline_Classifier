from Classifier_Code.back_import_clean import ImportAndClean
from Classifier_Code.back_train_get_accuracy_multinaivebayes import NBModelBuilder
from Classifier_Code.back_classify_multinaivebayes import NBClassifier
from Classifier_Code.back_train_get_accuracy_knn import KNNModelBuilder
from Classifier_Code.back_classify_knn import KNNClassifier

if __name__ == '__main__':

    # Check User Input
    # test_string = "Justin Bieber arrested after concert"
    test_string = "All the Times Ryan Reynolds And Blake Lively Roasted Each Other"
    # file_name = "sample_input.txt"

    # KNN
    # KNN Train
    importObject = ImportAndClean()
    text_column = importObject.get_text_column()
    numeric_column = importObject.get_label_column()

    # KNN Training and Testing
    knn_object_train = KNNModelBuilder(text_column, numeric_column, 0.1)
    print(knn_object_train.train())

    # KNN Classify
    # knn_object_classify = KNNClassifier()
    # print(knn_object_classify.buffer_for_prediction(
    #     importObject.clean_for_prediction(test_string, False)))

    # print(knn_object_classify.buffer_for_prediction(
    #     importObject.clean_for_prediction(file_name, True)))


    # Naive Bayes
    # Naive Bayes Train
    # importObject = ImportAndClean()
    # text_column = importObject.get_text_column()
    # numeric_column = importObject.get_label_column()
    #
    # # Naive Bayes Training and Testing
    # nb_object_train = NBModelBuilder(text_column, numeric_column, 0.1)
    # print(nb_object_train.train())

    ##Naive Bayes Classify
    # nb_object_classify = NBClassifier()

    # print(nb_object_classify.buffer_for_prediction(
    #     importObject.clean_for_prediction(file_name, True)))

    # print(nb_object_classify.buffer_for_prediction(
    #     importObject.clean_for_prediction(test_string, False)))