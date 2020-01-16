from tkinter import Tk, X, Y, TOP, BOTTOM, LEFT, RIGHT, BOTH, END, NO, W
from tkinter import Frame, Canvas, Button, Label, Entry, Checkbutton, IntVar
from tkinter.ttk import Treeview
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
import tkinter as tk
import time
import tkinter.ttk as ttk

import os
import random
from collections import namedtuple
from docplex.mp.model import Model

MAX_SQUARE_RATE = 2 / 3

MIN_COST = 1
MAX_COST = 10

root = Tk()
root.geometry("1200x800+300+100")

Square = namedtuple("Square", "height, width, cost, row, column")

def build_packing_nonoverlay_problem(rect_array, g):
    mdl = Model('nonoverlay_packing')

    nRectangle = len(rect_array)
    K = range(nRectangle)

    nRows = len(g)
    nCols = len(g[0])

    rows = range(nRows)
    cols = range(nCols)
    
    p = [[[0 for j in cols] for i in rows] for r in K]
    
    # calculate profit
    for r in K:
        w = rect_array[r].width
        h = rect_array[r].height
        c = rect_array[r].cost

        for i in rows :
            for j in cols :
                if i + h <= nRows and j + w <= nCols :
                    sum = 0
                    end_row = i + h
                    end_col = j + w
                    
                    for ii in range(i, end_row):
                        for jj in range(j, end_col):
                            sum += g[ii][jj]

                    p[r][i][j] = sum - c        

    # print(p)

    # declar var
    idx = [(r, i, j) for r in K for i in rows for j in cols ]
    mdl.x = mdl.binary_var_dict(idx, None, None, "X")
   
    # contraints

    # make sure that only one location is chosen for each rectangle
    mdl.add_constraints(mdl.sum(mdl.x[r, i, j] for i in rows for j in cols) <= 1 for r in K)

    # make sure that only one location is chosen for each rectangle
    mdl.add_constraints(mdl.sum(mdl.x[r, u, v] for r in K for u in rows for v in cols 
                        if u <= i and i < u + rect_array[r].height and v <= j and j < v + rect_array[r].width ) <= 1 
                        for i in rows for j in cols)

    mdl.add_constraints(mdl.x[r, i, j] == 0 for r in K for i in rows for j in cols if i + rect_array[r].height > nRows or j + rect_array[r].width > nCols )

    # objective values
    mdl.maximize(mdl.sum(mdl.x[r, i, j] * p[r][i][j] for r in K for i in rows for j in cols))
    
    return mdl;


def build_packing_overlay_problem(rect_array, g):
    mdl = Model('overlay_packing')

    nRectangle = len(rect_array)
    K = range(nRectangle)

    nRows = len(g)
    nCols = len(g[0])

    rows = range(nRows)
    cols = range(nCols)
    
    # declar var
    idx = [(r, i, j) for r in K for i in rows for j in cols ]
    mdl.x = mdl.binary_var_dict(idx, None, None, "X")

    idy = [(i, j) for i in rows for j in cols ]
    mdl.y = mdl.binary_var_dict(idy, None, None, "Y")
    
    # contraints

    # make sure that only one location is chosen for each rectangle
    mdl.add_constraints(mdl.sum(mdl.x[r, i, j] for i in rows for j in cols) <= 1 for r in K)

    # bigger
    mdl.add_constraints(mdl.sum(mdl.x[r, u, v] for r in K for u in rows for v in cols 
                        if u <= i and i < u + rect_array[r].height and v <= j and j < v + rect_array[r].width ) >= mdl.y[i, j] 
                        for i in rows for j in cols if g[i][j] > 0)

    # smaller
    mdl.add_constraints(mdl.sum(mdl.x[r, u, v] for r in K for u in rows for v in cols 
                        if u <= i and i < u + rect_array[r].height and v <= j and j < v + rect_array[r].width ) <= nRectangle * mdl.y[i, j] 
                        for i in rows for j in cols if g[i][j] < 0)   

    mdl.add_constraints(mdl.x[r, i, j] == 0 for r in K for i in rows for j in cols if i + rect_array[r].height > nRows or j + rect_array[r].width > nCols )

    # objective values
    mdl.maximize(mdl.sum(mdl.y[i, j] * g[i][j] for i in rows for j in cols) - mdl.sum(mdl.x[r, i, j] * rect_array[r].cost for r in K for i in rows for j in cols))
    
    return mdl;    

class Problem:
    def __init__(self):
        self.board = None
        self.squares = None
        
    def load(self, filename):
        with open(filename) as fp:
            lines = list(line for line in (l.strip() for l in fp) if line)
            
        if len(lines) < 2:
            messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
            return
        
        try:
            num_rows = int(lines[0])
            num_columns = int(lines[1])
        except:
            messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
            return
            
        if len(lines) < 2 + num_rows:
            messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
            return
        
        self.board = []
        for i in range(num_rows):
            row = lines[2 + i]
            elements = row.split(",")
            
            if len(elements) != num_columns:
                messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                return
            
            row = []
            for j in range(num_columns):
                try:
                    element = int(elements[j])
                except:
                    messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                    return
                row.append(element)
            self.board.append(row)
            
        if len(lines) <= 2 + num_rows:
            return
        
        try:
            num_squares = int(lines[2 + num_rows])
        except:
            messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
            return            
            
        if len(lines) < 3 + num_rows + num_squares:
            messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
            return
        
        self.squares = []
        for i in range(num_squares):
            line = lines[3 + num_rows + i]
            elements = line.split(",")
            
            if len(elements) != 3:
                messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                return
            
            try:
                height = int(elements[0])
            except:
                messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                return
            
            try:
                width = int(elements[1])
            except:
                messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                return
            
            try:
                cost = int(elements[2])
            except:
                messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                return
            
            square = Square(height, width, cost, -1, -1)
            self.squares.append(square)
            
        i = 1
                
    def save(self, filename):
        with open(filename, 'w') as fp:
            if self.board is None or self.squares is None:
                messagebox.showerror("No information", "The problem is not loaded yet.")
                return
            
            if len(self.board) == 0:
                messagebox.showerror("Invalid Problem", "The problem is not valid.")
                return
            
            if len(self.board[0]) == 0:
                messagebox.showerror("Invalid Problem", "The problem is not valid.")
                return
            
            num_rows = len(self.board)
            num_columns = len(self.board[0])
            
            fp.write(str(num_rows) + "\n")
            fp.write(str(num_columns) + "\n")
            
            for row in self.board:
                fp.write(",".join(map(str, row)) + "\n")
                
            if self.squares is None:
                return
            
            if len(self.squares) == 0:
                return
            
            num_squares = len(self.squares)
            fp.write(str(num_squares) + "\n")
            
            for row in self.squares:
                fp.write(str(row.height) + "," + str(row.width) + "," + str(row.cost) + "\n")
    
    def generate(self, class_num, num_rows, num_cols, num_squares, size_squares, chk_random):
        self.board = []
        count = 0
        sum1 = 0
        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                val = 1
                if class_num == 1 : # [0, 9]
                    val = random.randint(0, 9)
                elif class_num == 2 :   # [-9, 9]
                    val = random.randint(-9, 9)    
                elif class_num == 3 :   # [-10000, 0 - 9]
                    val = random.randint(0, 9)    
                    if random.randint(0, 100) < 5 :
                        val = -10000
                else :
                    val = random.randint(0, 1)       

                if val > -10000 :
                    sum1 += val 
                    count += 1;   

                row.append(val)

            self.board.append(row)
        
        max_height = max(int(num_rows * MAX_SQUARE_RATE), 2)
        max_width = max(int(num_cols * MAX_SQUARE_RATE), 2)
        
        min_height = 2
        min_width = 2

        avg = 0
        if count > 0 :
            avg = int(sum1 / count)
        
        self.squares = []
        for i in range(num_squares):
            if chk_random :
                height = random.randint(min_height, max_height)
                width = random.randint(min_width, max_width)
            else :
                height = size_squares
                width = size_squares

            # If we buy the squares (should be an option), the cost should be something which is not too
            # big or not too small comparing to the
            # gained values in the board cells
            min_cost = (avg - 1) * width * height  
            if min_cost < 1 :
                min_cost = 1

            max_cost = (avg + 1) * width * height   
            cost = random.randint(min_cost, max_cost)
            sqare = Square(height, width, cost, -1, -1)
            self.squares.append(sqare)
    
    def solve(self, overlap):
        if self.board is None or self.squares is None:
            messagebox.showerror("No information", "The problem is not loaded yet.")
            return
            
        if len(self.board) == 0:
            messagebox.showerror("Invalid Problem", "The problem is not valid.")
            return
        
        if len(self.board[0]) == 0:
            messagebox.showerror("Invalid Problem", "The problem is not valid.")
            return
        
        if len(self.squares) == 0:
            messagebox.showerror("Invalid Problem", "The problem is not valid.")
            return
        
        num_rows = len(self.board)
        num_columns = len(self.board[0])
        num_squares = len(self.squares)
        
        # Solve Algorithm - Will Change in future
        if overlap :
            model = build_packing_overlay_problem(self.squares, self.board)
        else :
            model = build_packing_nonoverlay_problem(self.squares, self.board)

        # model.parameters.threads = 10

        # Solve the model.
        if model.solve():
            obj = model.objective_value
            print("objective: {:g}".format(obj))
            for r in range(num_squares):
                square = self.squares[r]
                height = square.height
                width = square.width
                cost = square.cost

                row = -100; column = -100

                for i in range(num_rows) :
                    for j in range(num_columns):
                        if model.x[r, i, j].solution_value > 0 :
                            column = j
                            row = i

                self.squares[r] = Square(height, width, cost, row, column)
        else:
            print("Problem has no solution")        

        # for i in range(num_squares):
        #     square = self.squares[i]
        #     height = square.height
        #     width = square.width
        #     cost = square.cost
        #     row = random.randint(0, num_rows - height)
        #     column = random.randint(0, num_columns - width)
        #     self.squares[i] = Square(height, width, cost, row, column)
        
class ProblemWindow:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()

        self.num_rows = 0
        self.num_columns = 0
        self.num_squares = 0
        self.size_squares = 0
        self.chk_random = IntVar()

        self.status = False
        
        self.initUI()

    def initUI(self):
        top = self.top
        top.geometry("280x230+600+400")
        
        frmMain = Frame(top, padx = 10, pady = 5)
        frmMain.pack()

        frmNumInputClass = Frame(frmMain, pady = 5)
        frmNumInputClass.pack(fill=X)

        lblNumInputClass = Label(frmNumInputClass, text = "Input Class: ")
        lblNumInputClass.pack(side=LEFT)

        # entry_text = tk.StringVar()
        # self.txtfrmNumInputClass = Entry(frmNumInputClass, textvariable=entry_text)
        # self.txtfrmNumInputClass.pack(side=RIGHT)
        # entry_text.set("1")

        self.cb = ttk.Combobox(frmNumInputClass, values=("1", "2", "3", "4"))
        self.cb.set("1")
        self.cb.pack(side=RIGHT)
        # self.cb.bind('<<ComboboxSelected>>', on_select)
        
        frmNumRows = Frame(frmMain, pady = 5)
        frmNumRows.pack(fill=X)
        
        lblNumRows = Label(frmNumRows, text = "Number of rows: ")
        lblNumRows.pack(side=LEFT)
        
        self.txtNumRows = Entry(frmNumRows)
        self.txtNumRows.pack(side=RIGHT)
        
        frmNumCols = Frame(frmMain, pady = 5)
        frmNumCols.pack(fill=X)
        
        lblNumCols = Label(frmNumCols, text = "Number of columns: ")
        lblNumCols.pack(side=LEFT)
        
        self.txtNumCols = Entry(frmNumCols)
        self.txtNumCols.pack(side=RIGHT)
        
        frmNumSquares = Frame(frmMain, pady = 5)
        frmNumSquares.pack(fill=X)
        
        lblNumSquares = Label(frmNumSquares, text = "Number of squares: ")
        lblNumSquares.pack(side=LEFT)
        
        self.txtNumSquares = Entry(frmNumSquares)
        self.txtNumSquares.pack(side=RIGHT)

        frmSizeSquares = Frame(frmMain, pady = 5)
        frmSizeSquares.pack(fill=X)

        lblSizeSquares = Label(frmSizeSquares, text = "Size of squares: ")
        lblSizeSquares.pack(side=LEFT)

        self.txtSizeSquares = Entry(frmSizeSquares)
        self.txtSizeSquares.pack(side=RIGHT)

        frmChkRandom = Frame(frmMain, pady = 5)
        frmChkRandom.pack(fill=X)

        self.chkRandom = Checkbutton(frmChkRandom, text="Random Squares Size", variable=self.chk_random,command=self.onCheckRandom)
        self.chkRandom.pack(side=LEFT)
        
        frmButtons = Frame(frmMain, pady = 5)
        frmButtons.pack(fill = X, expand=True)

        btnOK = Button(frmButtons, text = "OK", width = 10, command = self.ok)
        btnOK.pack(side=LEFT, expand=True)
        
        btnCancel = Button(frmButtons, text = "Cancel", width = 10, command = self.cancel)
        btnCancel.pack(side=RIGHT, expand=True)

    def onCheckRandom(self):
        if self.chk_random.get():
            self.txtSizeSquares.config(state='disabled') 
        else:    
            self.txtSizeSquares.config(state='normal') 
        
    def ok(self):
        valid = True
        
        try:
            self.class_num = int(self.cb.get())
        except:
            valid = False

        try:
            self.num_rows = int(self.txtNumRows.get())
        except:
            valid = False
        
        try:
            self.num_columns = int(self.txtNumCols.get())
        except:
            valid = False
        
        try:
            self.num_squares = int(self.txtNumSquares.get())
        except:
            valid = False

        try:
            if self.chk_random.get():                
                self.size_squares = 0
            else:    
                self.size_squares = int(self.txtSizeSquares.get())                
        except:
            valid = False    
        
        if valid:
            self.status = True
            self.top.destroy()
        else:
            self.num_rows = 0
            self.num_columns = 0
            self.num_squares = 0            
        
    def cancel(self):
        self.status = False
        self.top.destroy()

class CellEditWindow:
    def __init__(self, parent, row_num, col_num, cell_value):
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()

        self.row_num = row_num
        self.col_num = col_num
        self.cell_value = cell_value
        
        self.status = False
        
        self.initUI()

    def initUI(self):
        top = self.top
        top.geometry("280x100+600+400")
        
        frmMain = Frame(top, padx = 10, pady = 5)
        frmMain.pack()

        frmCellValue = Frame(frmMain, pady = 5)
        frmCellValue.pack(fill=X)

        label_str = "({}, {}): ".format(self.row_num + 1, self. col_num + 1)
        lblCellValue = Label(frmCellValue, text = label_str)
        lblCellValue.pack(side=LEFT)
        
        entry_text = tk.StringVar()
        self.txtCellValue = Entry(frmCellValue, textvariable=entry_text)
        self.txtCellValue.pack(side=RIGHT)
        entry_text.set(self.cell_value)
        self.txtCellValue.focus()
                
        frmButtons = Frame(frmMain, pady = 5)
        frmButtons.pack(fill = X, expand=True)

        btnOK = Button(frmButtons, text = "OK", width = 10, command = self.ok)
        btnOK.pack(side=LEFT, expand=True)
        
        btnCancel = Button(frmButtons, text = "Cancel", width = 10, command = self.cancel)
        btnCancel.pack(side=RIGHT, expand=True)
        
    def ok(self):
        valid = True
        
        try:
            self.cell_value = int(self.txtCellValue.get())
        except:
            valid = False

        if valid:            
            self.status = True
            self.top.destroy()
        
    def cancel(self):
        self.status = False
        self.top.destroy()


class MainWindow(Frame):
    def __init__(self):
        super().__init__()
        
        self.problem = Problem()
        self.initUI()
        
    def initUI(self):
        self.master.title("Board Packing Problem")
        self.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        frmBoard = Frame(self, background="red")
        frmBoard.pack(fill=BOTH, expand=True)
        
        self.cnsBoard = Canvas(frmBoard, highlightthickness=1, highlightbackground="grey")
        self.cnsBoard.pack(fill=BOTH, expand=True, side=LEFT)
        self.cnsBoard.bind("<Key>", self.key)
        self.cnsBoard.bind("<Button-1>", self.callback)
        
        frmBoardSpace1 = Frame(frmBoard, width=10)
        frmBoardSpace1.pack(fill=BOTH, side=LEFT)
        
        self.tblSquare = Treeview(frmBoard, height=25)
        self.tblSquare.pack(fill=BOTH, side=RIGHT)
        
        self.tblSquare["columns"] = ("one" ,"two", "top", "left", "three", "four", "five")
        
        self.tblSquare.column("#0", width=40, stretch=NO)
        self.tblSquare.column("one", width=60, stretch=NO)
        self.tblSquare.column("two", width=60, stretch=NO)
        self.tblSquare.column("top", width=60, stretch=NO)
        self.tblSquare.column("left", width=60, stretch=NO)
        self.tblSquare.column("three", width=60, stretch=NO)
        self.tblSquare.column("four", width=60, stretch=NO)
        self.tblSquare.column("five", width=60, stretch=NO)
        
        self.tblSquare.heading("#0",text="No",anchor=W)
        self.tblSquare.heading("one", text="Height",anchor=W)
        self.tblSquare.heading("two", text="Width",anchor=W)
        self.tblSquare.heading("top", text="Top",anchor=W)
        self.tblSquare.heading("left", text="Left",anchor=W)
        self.tblSquare.heading("three", text="Cost",anchor=W) 
        self.tblSquare.heading("four", text="Board",anchor=W) 
        self.tblSquare.heading("five", text="Profit",anchor=W) 
        
        self.tblSquare.tag_configure("grey", background="grey")
        self.tblSquare.tag_configure("white", background="white")
                
        frmBoardSpace2 = Frame(self, height=10)
        frmBoardSpace2.pack(fill=BOTH)
        
        frmControl = Frame(self)
        frmControl.pack(fill=BOTH, expand=False)
        
        self.chk_overlap = IntVar()
        
        self.chkOverlap = Checkbutton(frmControl, text="Overlap", variable=self.chk_overlap)
        self.chkOverlap.pack(fill=Y, expand=False, side=LEFT)
        
        self.btnSolve = Button(frmControl, text="Solve Problem", width=22, command=self.solve_problem)
        self.btnSolve.pack(fill=Y, expand=False, side=RIGHT)
        
        frmBoardSpace3 = Frame(frmControl, width=14)
        frmBoardSpace3.pack(fill=BOTH, side=RIGHT)        
        
        self.btnGenerate = Button(frmControl, text="Generate Problem", width=22, command=self.generate_problem)
        self.btnGenerate.pack(fill=Y, expand=False, side=RIGHT)
        
        frmBoardSpace4 = Frame(frmControl, width=14)
        frmBoardSpace4.pack(fill=BOTH, side=RIGHT)
        
        self.btnExport = Button(frmControl, text="Export Problem", width=22, command=self.export_problem)
        self.btnExport.pack(fill=Y, expand=False, side=RIGHT)
        
        frmBoardSpace5 = Frame(frmControl, width=14)
        frmBoardSpace5.pack(fill=Y, expand=False, side=RIGHT)
        
        self.btnImport = Button(frmControl, text="Import Problem", width=22, command=self.import_problem)
        self.btnImport.pack(fill=Y, expand=False, side=RIGHT)
        
    def import_problem(self):
        filename = filedialog.askopenfilename(title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
        if not filename:
            return
        self.problem.load(filename)
        self.cnsBoard.delete("all")
        self.display_problem()        
        
    def export_problem(self):
        filename = filedialog.asksaveasfilename(title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
        if not filename:
            return        
        self.problem.save(filename)
    
    def generate_problem(self):
        problem_dialog = ProblemWindow(root)
        root.wait_window(problem_dialog.top)
        
        if not problem_dialog.status:
            return
        
        class_num = problem_dialog.class_num
        num_rows = problem_dialog.num_rows
        num_columns = problem_dialog.num_columns
        num_squares = problem_dialog.num_squares
        size_squares = problem_dialog.size_squares
        chk_random = problem_dialog.chk_random.get()

        self.problem.generate(class_num, num_rows, num_columns, num_squares, size_squares, chk_random)
        self.cnsBoard.delete("all")
        self.display_problem()
    
    def solve_problem(self):
        overlap = self.chk_overlap.get()
        start = time.time()
        self.problem.solve(overlap)
        end = time.time()
        self.display_problem(solution=True)

        gap = (end - start)
        elapsed = time.strftime('%H:%M:%S', time.gmtime(gap))
        self.master.title("Board Packing Problem" + " ------ Running Time = " + elapsed)
        
    def display_problem(self, solution = False):
        canvas = self.cnsBoard
        board = self.problem.board
        squares = self.problem.squares
        
        if board is None or len(board) == 0 or len(board[0]) == 0:
            return
        
        num_rows = len(board)
        num_columns = len(board[0])

        canvas_height = canvas.winfo_height()        
        canvas_width = canvas.winfo_width()
        
        cell_size = min(canvas_height / num_rows, canvas_width / num_columns)
        board_height = cell_size * num_rows
        board_width = cell_size * num_columns
        
        board_top = (canvas_height - board_height) / 2
        board_left = (canvas_width - board_width) / 2
        
        canvas.create_rectangle(board_left, board_top, board_left + board_width, board_top + board_height, fill="white", outline="")
        
        for i in range(num_rows + 1):
            line_top = i * cell_size + board_top
            canvas.create_line(board_left, line_top, board_left + board_width, line_top, fill="lightgrey")
            
        for i in range(num_columns + 1):
            line_left = i * cell_size + board_left
            canvas.create_line(line_left, board_top, line_left, board_top + board_height, fill="lightgrey")
    
        for i in range(num_rows):
            for j in range(num_columns):
                element = board[i][j]
                if element != 0:
                    rect_top = i * cell_size + board_top
                    rect_left = j * cell_size + board_left
                    rect_bottom = rect_top + cell_size
                    rect_right = rect_left + cell_size                                        
                    if element > 0 :
                        canvas.create_rectangle(rect_left, rect_top, rect_right, rect_bottom, fill="lightgrey", outline="")
                    elif element > -10000 :    
                        canvas.create_rectangle(rect_left, rect_top, rect_right, rect_bottom, fill="darkgrey", outline="")
                    else :
                        canvas.create_rectangle(rect_left, rect_top, rect_right, rect_bottom, fill="darkred", outline="")


        if solution:
            num = 0
            for square in squares:
                height = square.height
                width = square.width
                cost = square.cost
                row = square.row
                column = square.column
                
                square_top = row * cell_size + board_top
                square_left = column * cell_size + board_left
                square_bottom = square_top + height * cell_size
                square_right = square_left + width * cell_size
                
                fill = "#" + ("%06x" % random.randint(0, 16777215))
                canvas.create_rectangle(square_left, square_top, square_right, square_bottom, fill=fill, outline="")

                self.tblSquare.tag_configure(str(num + 1), background=fill)

                num += 1
                
                    
        table = self.tblSquare
        table.delete(*table.get_children())
 
        total_cost = 0
        total_value = 0
        total_profit = 0
        total_count = 0
        for (key, square) in enumerate(squares):
            value = ""
            sum1 = 0
            if square.row >= 0 and square.column >= 0: 
                for i in range(square.row, square.row + square.height):
                    for j in range(square.column, square.column + square.width):
                        sum1 += board[i][j]

                value = sum1        

            if value == "":
                profit = ""
            else:
                profit = value - square.cost    
                total_cost += square.cost 
                total_value += value
                total_profit += profit
            
            if solution:
                tags = (str(key + 1),)
            else:                
                if key % 2 == 0:
                    tags = ("white",)
                else:
                    tags = ("grey",)                    

            table.insert("", "end", None, text=str(key + 1), values=(str(square.height), str(square.width), str(square.row + 1), str(square.column + 1), str(square.cost), str(value), str(profit)), tags=tags)

            total_count += 1

        table.insert("", "end", None, text=str(total_count + 1), values=("Total", "", "", "", str(total_cost), str(total_value), str(total_profit)), tags=("green",))
        
        
                
        for i in range(num_rows):
            for j in range(num_columns):
                element = board[i][j]
                rect_top = i * cell_size + board_top
                rect_left = j * cell_size + board_left                
                text_top = rect_top + cell_size / 2
                text_left = rect_left + cell_size / 2
                canvas.create_text(text_left, text_top, text = str(element))

    def key(self, event):
        print("pressed", repr(event.char))

    def callback(self, event):    
        canvas = self.cnsBoard
        board = self.problem.board
        squares = self.problem.squares
        
        if board is None or len(board) == 0 or len(board[0]) == 0:
            return
        
        num_rows = len(board)
        num_columns = len(board[0])

        canvas_height = canvas.winfo_height()        
        canvas_width = canvas.winfo_width()
        
        cell_size = min(canvas_height / num_rows, canvas_width / num_columns)
        board_height = cell_size * num_rows
        board_width = cell_size * num_columns
        
        board_top = (canvas_height - board_height) / 2
        board_left = (canvas_width - board_width) / 2
        
        col_num = int((event.x - board_left) / cell_size)
        row_num = int((event.y - board_top) / cell_size)
        if col_num < 0 or col_num >= num_columns :
            return;
        if row_num < 0 or row_num >= num_rows :
            return;

        # display cell edit dialog
        cell_edit_dialog = CellEditWindow(root, row_num, col_num, board[row_num][col_num])
        
        root.wait_window(cell_edit_dialog.top)
        
        if not cell_edit_dialog.status:
            return
        
        cell_value = cell_edit_dialog.cell_value
        board[row_num][col_num] = cell_value
        self.display_problem()

        print(row_num, col_num);

                
app = MainWindow()

root.mainloop()