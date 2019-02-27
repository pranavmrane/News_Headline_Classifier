from tkinter import *
from tkinter import messagebox
from Classifier_Code.back_import_clean import ImportAndClean
from Classifier_Code.back_classify_knn import KNNClassifier
from tkinter.filedialog import askopenfilename
import time
import datetime


class KNNClassificationDisplay:

    def __init__(self):
        self.window = Tk()
        self.window.title("KNN Classification Window")
        background_color = "black"
        text_color = "white"
        self.window.configure(background=background_color)
        self.window.resizable(width=False, height=False)
        self.window.geometry('{}x{}'.format(450, 650))
        self.row_number = 0
        self.headline_style = "none 20 bold"
        self.sub_headline_style = "none 16 bold"
        self.regular_text_style = "none 12"
        self.foot_text_style = "none 8"
        Label(self.window, text="Real-Time Headline Classifier",
              bg=background_color, fg=text_color,font=self.headline_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        Label(self.window, text="KNN - Classification", bg=background_color,
              fg=text_color, font=self.sub_headline_style).\
            grid(row=self.get_incremented_row(),column=0, sticky=W)
        Label(self.window, text="\nEnter Headline to be Classified*",
              bg=background_color, fg=text_color, font=self.regular_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        self.textentry = Entry(self.window, width=44, bg=text_color,
                               font=self.regular_text_style)
        self.textentry.grid(row=self.get_incremented_row(), column=0, sticky=W)
        Button(self.window, text="SUBMIT", width=8,
               command=self.classify_button_single).\
            grid(row=self.get_incremented_row(), column=0, sticky=E)

        Label(self.window, text="\nOR",bg=background_color,
              fg=text_color, font=self.sub_headline_style).grid(row=self.get_incremented_row(),
                                               column=0, sticky=W)

        Label(self.window, text="\nChoose File with Headlines to be Classified*",
              bg=background_color,
              fg=text_color, font=self.regular_text_style).grid(row=self.get_incremented_row(),
                                               column=0, sticky=W)
        self.address_display = Entry(self.window, width=44, bg=text_color,
                                     font=self.regular_text_style)
        self.address_display.grid(row=self.get_incremented_row(), column=0,
                                  sticky=W)
        Button(self.window, text="CHOOSE FILE", width=8,
               command=self.load_file).grid(row=self.get_incremented_row(),
                                            column=0, sticky=W)
        Button(self.window, text="SUBMIT", width=8,
               command=self.classify_button_multiple).grid(
            row=self.get_row(), column=0, sticky=E)

        Label(self.window, text="\nHeadlines and their respective class:**",
              bg=background_color, fg=text_color, font=self.regular_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)

        self.output = Text(self.window, width=44, height=10, wrap=WORD,
                      background=text_color, font=self.regular_text_style)
        self.output.grid(row=self.get_incremented_row(), column=0,
                         columnspan=2, sticky=W)
        Label(self.window,
              text="\n* - Compulsory Fields",
              bg=background_color, fg=text_color, font=self.foot_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        Label(self.window,
              text="\n** - Results Stored also stored as a Text File in Project Folder",
              bg=background_color, fg=text_color, font=self.foot_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        Button(self.window, text="EXIT", width=8,
               command=self.close_window).grid(row=self.get_incremented_row(),
                                               column=0, sticky=W)
        Button(self.window, text="CLEAN", width=8,
               command=self.clear_text_boxes).grid(row=self.get_row(),
                                               column=0, sticky=E)

        self.window.mainloop()

    def classify_button_single(self):
        entered_text = self.textentry.get()
        # print(entered_text)
        if len(entered_text) < 1:
            messagebox.showinfo("Error", "Please enter a headline")
        else:
            self.output.delete(0.0, END)
            importObject = ImportAndClean()
            simple_predictor = KNNClassifier()
            # Check User Input
            # test_string = "Justin Bieber arrested after concert"
            result = simple_predictor.buffer_for_prediction(
                importObject.clean_for_prediction(entered_text, False))
            self.save_results(result)
            self.output.insert(0.0, result)

    def classify_button_multiple(self):
        address = self.address_display.get()
        if len(address) < 1:
            messagebox.showinfo("Error", "Please choose a dataset")
        else:
            importObject = ImportAndClean()
            simple_predictor = KNNClassifier()
            # Check User Input
            file_name = "sample_input.txt"
            self.output.delete(0.0, END)
            result = simple_predictor.buffer_for_prediction(
                importObject.clean_for_prediction(address, True))
            self.save_results(result)
            self.output.insert(0.0, result)

    def clear_text_boxes(self):
        self.output.delete(0.0, END)
        self.textentry.delete(0, END)
        self.address_display.delete(0, END)

    def save_results(self, string_to_stored):
        file_name = "results" + datetime.datetime.fromtimestamp(time.time()).\
            strftime('%Y-%m-%d %H:%M:%S') + ".txt"
        #os.mknod(file_name)
        with open(file_name, "w") as text_file:
            text_file.write(string_to_stored)

    def get_incremented_row(self):
        self.row_number += 1
        return self.row_number

    def get_row(self):
        return self.row_number

    def load_file(self):
        fname = askopenfilename(filetypes=(("Txt files", "*.txt"),
                                           ("All files", "*.*")))
        self.address_display.delete(0, END)
        self.address_display.insert(0, fname)

    def close_window(self):
        self.window.destroy()
        exit()


if __name__ == '__main__':
    value = KNNClassificationDisplay()
