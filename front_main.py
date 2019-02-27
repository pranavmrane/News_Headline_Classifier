from tkinter import *
from UI_Files.front_train_multinaivebayes import NBTrainDisplay
from UI_Files.front_classify_multinaivebayes import NBClassificationDisplay
from UI_Files.front_train_knn import KNNTrainDisplay
from UI_Files.front_classify_knn import KNNClassificationDisplay

class MainDisplay:

    def __init__(self):
        self.window = Tk()
        self.window.title("Choose Options")
        background_color = "black"
        text_color = "white"
        self.window.configure(background=background_color)
        self.window.resizable(width=False, height=False)
        self.window.geometry('{}x{}'.format(450, 300))
        self.row_number = 0
        self.headline_style = "none 20 bold"
        self.sub_headline_style = "none 16 bold"
        self.regular_text_style = "none 12"
        Label(self.window, text="Real-Time Headline Classifier",
              bg=background_color, fg=text_color, font=self.headline_style).pack()
        Label(self.window, text="Landing Page", bg=background_color,
              fg=text_color, font=self.sub_headline_style).pack()
        Label(self.window, text="Choose One of the following",
              bg=background_color, fg=text_color, font="none 12 bold").pack()
        Label(self.window, text="Multinomial Naive Bayes",
              bg=background_color, fg=text_color, font="none 12 bold").pack()
        Button(self.window, text="TRAIN", width=8, command=self.load_train_nb).pack()
        Button(self.window, text="CLASSIFY", width=8,
               command=self.load_classify_nb).pack()
        Label(self.window, text="K - Nearest Neighbours",
              bg=background_color, fg=text_color, font="none 12 bold").pack()
        Button(self.window, text="TRAIN", width=8, command=self.load_train_knn).pack()
        Button(self.window, text="CLASSIFY", width=8,
               command=self.load_classify_knn).pack()
        Button(self.window, text="EXIT", width=8, command=self.close_window).pack()

        self.window.mainloop()

    def load_train_nb(self):
        self.window.destroy()
        value = NBTrainDisplay()

    def load_classify_nb(self):
        self.window.destroy()
        value = NBClassificationDisplay()

    def load_train_knn(self):
        self.window.destroy()
        value = KNNTrainDisplay()

    def load_classify_knn(self):
        self.window.destroy()
        value = KNNClassificationDisplay()

    def close_window(self):
        self.window.destroy()
        exit()

    def get_incremented_row(self):
        self.row_number += 1
        return self.row_number

if __name__ == '__main__':
    value = MainDisplay()
