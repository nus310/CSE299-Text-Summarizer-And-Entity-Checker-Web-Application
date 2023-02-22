# import subprocess
# try:
#    import pyautogui
# except ImportError:
#     subprocess.call("pip install pyautogui")

import pyautogui
import tkinter as tk
from tkinter.filedialog import *

root = tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 300)
canvas1.pack()

def takeScreenshot():
    try:
        mySS = pyautogui.screenshot()
        save_path = asksaveasfilename()
        mySS.save(save_path, "screenshot.png")
    except ValueError:
        print("please sss")
button = tk.Button(text = "take Screenshot", command = takeScreenshot(), font = 10)
canvas1.create_window(150, 150 , window = button)
root.mainloop()


# import PyPDF2
# a = PyPDF2.PdfFileReader('ELES.pdf')
# str= ""

# for i in range(1,5):
#     str += a.getPage(i).extractText()

# with open("text.txt", "w" ) as f:
#     f.write(str)

