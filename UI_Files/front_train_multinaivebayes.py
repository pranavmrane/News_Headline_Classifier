from tkinter import *
from tkinter import messagebox
from Classifier_Code.back_import_clean import ImportAndClean
from Classifier_Code.back_train_get_accuracy_multinaivebayes import NBModelBuilder
from tkinter.filedialog import askopenfilename


class NBTrainDisplay:

    def __init__(self):
        self.window = Tk()
        self.window.title("Training Window")
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
        Label(self.window, text="Naive Bayes - Training Module", bg=background_color,
              fg=text_color, font=self.sub_headline_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        Label(self.window, text="\nSelect Dataset*",
              bg=background_color, fg=text_color, font=self.regular_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        self.address_display = Entry(self.window, width=44, bg="white",
                                     font=self.regular_text_style)
        self.address_display.grid(row=self.get_incremented_row(), column=0,
                                  sticky=W)
        Button(self.window, text="CHOOSE FILE", width=8,
               command=self.display_dataset).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)

        Label(self.window, text="\nEnter Test Size Ratio (Under 0.3)*",
              bg=background_color, fg=text_color, font=self.regular_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        self.textentry = Entry(self.window, width=44, bg="white",
                               font=self.regular_text_style)
        self.textentry.grid(row=self.get_incremented_row(), column=0, sticky=W)

        Button(self.window, text="TRAIN", width=8, command=self.train_button).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        Label(self.window,
              text="\nAccuracy and Confusion Matrix (Wait 6-8 Minutes)",
              bg=background_color, fg=text_color, font=self.regular_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)

        self.output = Text(self.window, width=44, height=12, wrap=WORD,
                      background="white", font=self.regular_text_style)
        self.output.grid(row=self.get_incremented_row(), column=0,
                         columnspan=2, sticky=W)
        Label(self.window,
              text="\n* - Compulsory Fields",
              bg=background_color, fg=text_color, font=self.foot_text_style).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        Button(self.window, text="EXIT", width=8, command=self.close_window).\
            grid(row=self.get_incremented_row(), column=0, sticky=W)
        Button(self.window, text="CLEAN", width=8,
               command=self.clear_text_boxes).grid(row=self.get_row(),
                                               column=0, sticky=E)

        self.window.mainloop()

    def get_incremented_row(self):
        self.row_number += 1
        return self.row_number

    def display_dataset(self):
        fname = askopenfilename(filetypes=(("CSV files", "*.csv"),
                                           ("All files", "*.*")))
        # print(fname)
        self.address_display.delete(0, END)
        self.address_display.insert(0, fname)

    def train_button(self):
        entered_text = self.textentry.get()
        # print(entered_text)

        if len(entered_text) == 0:
            messagebox.showinfo("Error", "Please enter a relevant value")
        elif float(entered_text) > 0.31:
            messagebox.showinfo("Error", "Enter a value smaller than 0.3")
        else:
            self.output.delete(0.0, END)
            if len(self.address_display.get()) < 1:
                messagebox.showinfo("Error", "Please choose a dataset")
            else:
                importObject = ImportAndClean(self.address_display.get())
                text_column = importObject.get_text_column()
                numeric_column = importObject.get_label_column()
                trainer = NBModelBuilder(text_column, numeric_column,
                                       float(entered_text))
                self.output.insert(END, trainer.train())

    def clear_text_boxes(self):
        self.output.delete(0.0, END)
        self.textentry.delete(0, END)
        self.address_display.delete(0, END)

    def close_window(self):
        self.window.destroy()
        exit()

    def get_row(self):
        return self.row_number

if __name__ == '__main__':
    value = NBTrainDisplay()

