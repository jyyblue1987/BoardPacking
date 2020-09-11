from tkinter import Tk, X, Y, TOP, BOTTOM, LEFT, RIGHT, BOTH, END, NO, W
from tkinter import Frame, Canvas, Button, Label, Entry, Checkbutton, IntVar
from tkinter.ttk import Treeview
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
import tkinter as tk
import time
import tkinter.ttk as ttk

from copy import copy, deepcopy

import os
import random
import colorsys
from collections import namedtuple
from os import listdir
from os.path import isfile, join

import warnings

try:
    from docplex.mp.model import Model
except ImportError:
    print('CPLEX Moudle is not installed')


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


class SquareIndividual(object): 
    ''' 
    Class representing individual in population 
    '''
    def __init__(self, chromosome, board, squares):         
        self.board = board  
        self.squares = squares          
        self.num_rows = len(self.board)
        self.num_cols = len(self.board[0])

        gnome_len = len(self.squares) 
        if len(chromosome) < 1 :
            chromosome = []

            for i in range(gnome_len):            
                chromosome.append(self.mutated_genes(i, 1)) # row

            for i in range(gnome_len):            
                chromosome.append(self.mutated_genes(i, 2)) # col    

        self.chromosome = chromosome

        self.fitness = self.cal_fitness() 
    
    def mutated_genes(self, num, flag): 
        ''' 
        create random genes for mutation 
        '''
        square = self.squares[num]
        if flag == 1: # row                         
            gene = random.randint(-2, self.num_rows - square.height)
        else:    
            gene = random.randint(-2, self.num_cols - square.width)

        return gene 
  
    def create_gnome(self): 
        ''' 
        create chromosome or string of genes 
        '''
        gnome_len = len(self.squares) 
        chromosome = []

        for i in range(gnome_len):            
            chromosome.append(self.mutated_genes(i, 1)) # row

        for i in range(gnome_len):            
            chromosome.append(self.mutated_genes(i, 2)) # col    

        return chromosome
  
    def mate(self, par2, muta_prob): 
        ''' 
        Perform mating and produce new offspring 
        '''
        crit_prob = (1 - muta_prob) / 2

        # chromosome for offspring 
        num_squares = len(self.squares)

        child_chromosome = [] 
        num = 0
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):     
  
            # random probability   
            prob = random.random() 
  
            # if prob is less than 0.45, insert gene 
            # from parent 1  
            if prob < crit_prob: 
                child_chromosome.append(gp1) 
  
            # if prob is between 0.45 and 0.90, insert 
            # gene from parent 2 
            elif prob < crit_prob * 2: 
                child_chromosome.append(gp2) 
  
            # otherwise insert random gene(mutate),  
            # for maintaining diversity 
            else: 
                if num >= num_squares:
                    child_chromosome.append(self.mutated_genes(num - num_squares, 2))
                else:    
                    child_chromosome.append(self.mutated_genes(num, 1))

            num += 1    
  
        # create new Individual(offspring) using  
        # generated chromosome for offspring 
        return SquareIndividual(child_chromosome, self.board, self.squares) 

    def crossover(self, par2): 
        # chromosome for offspring 
        num_squares = len(self.squares)

        child_chromosome = [] 
        num = 0
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):       
            # random probability   
            prob = random.random() 
  
            # if prob is less than 0.5, insert gene 
            # from parent 1  
            if prob < 0.5: 
                child_chromosome.append(gp1) 
  
            # if prob is between 0.5 and 1.0, insert 
            # gene from parent 2 
            else: 
                child_chromosome.append(gp2) 
  
        # create new Individual(offspring) using  
        # generated chromosome for offspring 
        return SquareIndividual(child_chromosome, self.board, self.squares) 

    def _mutate(self): 
        num = random.randrange(0, len(self.chromosome))        
        # chromosome for offspring 
        num_squares = len(self.squares)

        gene = -1
        if num >= num_squares:
            gene = self.mutated_genes(num - num_squares, 2)
        else:    
            gene = self.mutated_genes(num, 1)

        self.chromosome[num] = gene     

        self.fitness = self.cal_fitness() 
        
        
    def cal_fitness(self): 
        ''' 
        Calculate fittness score, it is the number of 
        characters in string which differ from target 
        string. 
        '''
        fitness = 0
      
        num_rows = len(self.board)
        num_columns = len(self.board[0])
        num_squares = len(self.squares)

        flag = [[0 for i in range(num_columns)] for j in range(num_rows)]                              
        total_cost = 0
        for r in range(num_squares):
            square = self.squares[r]
            height = square.height
            width = square.width
            cost = square.cost

            col = self.chromosome[r + num_squares]
            row = self.chromosome[r]

            if col <= -1 or row <= -1:    # dummy rectangle
                continue

            total_cost += cost

            for i in range(row, row + height):
                for j in range(col, col + width):
                    flag[i][j] = 1                                            

        # calulate total profit
        sum1 = 0
        for i in range(num_rows):
            row = []
            for j in range(num_columns):                                       
                if flag[i][j] > 0 :
                    sum1 += self.board[i][j]

        profit = sum1 - total_cost
        fitness = -profit

        return fitness 

class Problem:
    def __init__(self):
        self.board = None
        self.squares = None
        self.obj_val = 0
        self.class_num = 1
        
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
            
            if len(elements) < 3:
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

            row = -99
            col = -99
            if len(elements) >= 5:
                try:
                    row = int(elements[3])
                    if row < 0:
                        row = -99
                except:
                    messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                    return    

                try:
                    col = int(elements[4])
                    if col < 0:
                        col = -99
                except:
                    messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                    return            
            
            square = Square(height, width, cost, row, col)
            self.squares.append(square)
            
        self.obj_val = -1000000
        if len(lines) > 3 + num_rows + num_squares:
            try:
                self.obj_val = int(lines[3 + num_rows + num_squares])                
            except:
                messagebox.showerror("Invalid File Format", "The input file is invalid or corrupted.")
                return
                
        i = 1

        self.class_num = 1
                
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
                fp.write(str(row.height) + "," + str(row.width) + "," + str(row.cost) + "," + str(row.row) + "," + str(row.column) + "\n")

            if self.obj_val > -1000000:
                fp.write(str(self.obj_val) + "\n")     
    
    def generate(self, class_num, num_rows, num_cols, num_squares, size_squares, chk_random, chk_cost_random, cost_array):
        self.board = []
        self.class_num = class_num 
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
            if chk_cost_random :
                cost = random.randint(min_cost, max_cost)
            else :   
                cost = cost_array[i]

            sqare = Square(height, width, cost, -1, -1)
            self.squares.append(sqare)

    def refreshValues(self):
        num_rows = len(self.board)
        num_cols = len(self.board[0]) 
        self.board = []
        class_num = self.class_num
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

                row.append(val)

            self.board.append(row)

    def checkProblem(self):
        if self.board is None or self.squares is None:
            messagebox.showerror("No information", "The problem is not loaded yet.")
            return False
            
        if len(self.board) == 0:
            messagebox.showerror("Invalid Problem", "The problem is not valid.")
            return False
        
        if len(self.board[0]) == 0:
            messagebox.showerror("Invalid Problem", "The problem is not valid.")
            return False
        
        if len(self.squares) == 0:
            messagebox.showerror("Invalid Problem", "The problem is not valid.")
            return False

        return True    
    
    def solve(self, overlap):
        if self.checkProblem() == False:
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
        if model.solve(log_output=True):
            obj = model.objective_value
            self.obj_val = obj
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

    def checkOverlap(self, xx, yy, squares):
        num_squares = len(squares)

        for r in range(0,num_squares - 1):
            s1 = squares[r]
            h1 = s1.height
            w1 = s1.width
            y1 = yy[r] - 1
            x1 = xx[r] - 1
            b1 = y1 + h1
            r1 = x1 + w1

            if x1 < 0:
                continue

            for p in range(r + 1, num_squares):
                s2 = squares[p]
                h2 = s2.height
                w2 = s2.width
                x2 = xx[p] - 1
                y2 = yy[p] - 1
                b2 = y2 + h2
                r2 = x2 + w2

                if x2 < 0:
                    continue

                # check overlay
                if (r1 <= x2 or r2 <= x1) or (b1 <= y2 or b2 <= y1): # non overlay
                    continue
                else:
                    return True

        return False

    def calc_profit(self, board, squares, xx, yy):
        num_rows = len(board)
        num_columns = len(board[0])
        num_squares = len(squares)

        flag = [[0 for i in range(num_columns)] for j in range(num_rows)]                              
        total_cost = 0
        for r in range(num_squares):
            square = squares[r]
            height = square.height
            width = square.width
            cost = square.cost

            col = xx[r] - 1
            row = yy[r] - 1
            if col <= -1:    # dummy rectangle
                continue

            total_cost += cost

            for i in range(row, row + height):
                for j in range(col, col + width):
                    flag[i][j] = 1                                            

        # calulate total profit
        sum1 = 0
        for i in range(num_rows):
            row = []
            for j in range(num_columns):                                       
                if flag[i][j] > 0 :
                    sum1 += board[i][j]

        profit = sum1 - total_cost

        return profit

    def generate_squares(self, board, squares, xx, yy):    
        num_squares = len(squares)
        for r in range(num_squares):
            square = squares[r]
            height = square.height
            width = square.width
            cost = square.cost

            if xx[r] < 1: 
                row = -100; column = -100
            else:
                row = yy[r] - 1; column = xx[r] - 1

            squares[r] = Square(height, width, cost, row, column)     

        return squares 

    def solve_by_brute_force(self, overlap):
        if self.checkProblem() == False:
            return

        max_profit = -1000000
        # init rectangle with first position
        num_rows = len(self.board)
        num_columns = len(self.board[0])
        num_squares = len(self.squares)

        xx = [0]*num_squares
        pos_x = num_squares - 1
        xx[num_squares - 1] = -1

        total_possible_count = 0

        while True:
            if pos_x < 0:
                break

            if xx[pos_x] <= num_columns - self.squares[pos_x].width:
                xx[pos_x] += 1
                if pos_x < num_squares - 1:
                    pos_x += 1
                else:
                    # search row
                    yy = [1]*num_squares
                    pos_y = num_squares - 1
                    yy[num_squares - 1] = 0

                    while True:
                        if pos_y < 0:
                            break

                        if yy[pos_y] <= num_rows - self.squares[pos_y].height:
                            yy[pos_y] += 1
                            if pos_y < num_squares - 1:
                                pos_y += 1
                            else:
                                print(xx, yy)
                                # if xx[1] == 3 and yy[1] == 6:
                                #      xx[1] = 3
                                if overlap == False and self.checkOverlap(xx, yy, self.squares) == True :                                    
                                    continue

                                total_possible_count += 1

                                profit = self.calc_profit(self.board, self.squares, xx, yy)

                                if profit > max_profit:
                                    # set max
                                    max_profit = profit    
                                    self.obj_val = profit
                                    self.squares = self.generate_squares(self.board, self.squares, xx, yy)                                    
                        else:
                            yy[pos_y] = 0
                            pos_y -= 1
            else:
                xx[pos_x] = -1
                pos_x -= 1

        print('Total Brute Force Possible Solution Count = ', total_possible_count)

    def sort_by_width(self, val): 
        return val.width  

    def sort_by_height(self, val): 
        return val.height      

    def sort_by_area(self, val): 
        return val.width * val.height      

    def sort_by_cost(self, val): 
        return val.cost          

    def sort_by_area_cost(self, val): 
        return val.width * val.height * val.cost              

    def solve_by_greedy_proc(self, board, squares, overlap, option):
        b_h = len(board)
        b_w = len(board[0])
        num_squares = len(squares)

        sol_squares = squares
        obj_val = -1000000

        # sort rectangle 
        if option == 'random':
            random.shuffle(squares)  

        if option == 'width':
            squares.sort(key = self.sort_by_width, reverse = True)  

        if option == 'height':
            squares.sort(key = self.sort_by_height, reverse = True)  

        if option == 'area':
            squares.sort(key = self.sort_by_area, reverse = True)      

        if option == 'cost':
            squares.sort(key = self.sort_by_cost, reverse = True)          

        if option == 'area_cost':
            squares.sort(key = self.sort_by_area_cost, reverse = True)          

        max_profit = 0
        xx = [0]*num_squares
        yy = [0]*num_squares

        # init rectangle
        squares = self.generate_squares(board, squares, xx, yy)
        
        iterate_count = 0
        for r in range(num_squares):
            # select a rectangle
            square = squares[r]
            height = square.height
            width = square.width
            cost = square.cost

            sub_max_profit = -1000000
            pos_x = -1
            pos_y = -1

            # brute forth
            for i in range(b_h - height + 1):
                yy[r] = i + 1
                for j in range(b_w - width + 1):
                    xx[r] = j + 1
                    if overlap == False and self.checkOverlap(xx, yy, squares) == True :                                    
                        continue

                    iterate_count += 1

                    profit = self.calc_profit(board, squares, xx, yy)

                    if profit > sub_max_profit:
                        # set max
                        sub_max_profit = profit   
                        pos_x = j
                        pos_y = i

            if pos_x >= 0 and pos_y >= 0:
                xx[r] = pos_x + 1
                yy[r] = pos_y + 1

                profit = self.calc_profit(self.board, squares, xx, yy)
                if profit > max_profit:
                    # set max
                    max_profit = profit    
                    obj_val = profit
                    sol_squares = self.generate_squares(board, squares, xx, yy) 

                # change the covered g-values to zero
                for i in range(pos_y, pos_y + height):
                    for j in range(pos_x, pos_x + width):
                        board[i][j] = 0   

        return sol_squares, obj_val        

    def solve_by_local_search(self, board, squares, overlap, option, obj_val):
        sol_squares = squares

        # local search
        num_squares = len(squares)
        b_h = len(board)
        b_w = len(board[0])
        for r in range(num_squares):
            # pack a rectangle
            
            sub_squares = []            
            for q in range(num_squares):
                if r == q:
                    continue

                square = squares[q]
                height = square.height
                width = square.width
                cost = square.cost
  
                square1 = Square(height, width, cost, -100, -100)
                sub_squares.append(square1)

            board = deepcopy(self.board)
            sub_squares, profit = self.solve_by_greedy_proc(board, sub_squares, overlap, option)
           
            xx = [0]*num_squares
            yy = [0]*num_squares
            for q in range(num_squares - 1):
                square = sub_squares[q]
                xx[q] = square.column + 1
                yy[q] = square.row + 1

            # brute forth

            # append current rectangle to last 
            square = squares[r]
            height = square.height
            width = square.width
            cost = square.cost

            square1 = Square(height, width, cost, -100, -100)
            sub_squares.append(square1)

            pos_x = -1
            pos_y = -1
            q = num_squares - 1

            board = deepcopy(self.board)
            for i in range(b_h - height + 1):
                yy[q] = i + 1
                for j in range(b_w - width + 1):
                    xx[q] = j + 1
                    if overlap == False and self.checkOverlap(xx, yy, sub_squares) == True :                                    
                        continue

                    profit = self.calc_profit(board, sub_squares, xx, yy)

                    if profit > obj_val:
                        # set max
                        obj_val = profit   
                        pos_x = j
                        pos_y = i

            if pos_x >= 0 and pos_y >= 0:
                xx[q] = pos_x + 1
                yy[q] = pos_y + 1
                sol_squares = self.generate_squares(board, sub_squares, xx, yy)                

        return sol_squares, obj_val        

    def solve_by_greedy(self, overlap, option, local_search):
        if self.checkProblem() == False:
            return

        board = deepcopy(self.board)
        squares = deepcopy(self.squares)

        sol_squares = squares
        sol_squares, obj_val = self.solve_by_greedy_proc(board, squares, overlap, option)

        if local_search:
            sol_squares, obj_val = self.solve_by_local_search(board, squares, overlap, option, obj_val)

        self.squares = sol_squares
        self.obj_val = obj_val

    def solve_by_ga(self, overlap, popu_size, stop_size, muta_prob, select_prob):
        board = deepcopy(self.board)
        squares = deepcopy(self.squares)

        # Number of individuals in each generation 
        POPULATION_SIZE = popu_size

        #current generation 
        generation = 1
    
        found = False
        population = [] 

        min_fitness = 1000000000
        min_generation_num = 0
    
        # create initial population 
        for _ in range(POPULATION_SIZE): 
            population.append(SquareIndividual([], board, squares)) 

        muta_prob1 = muta_prob
        chromosome = []
        while not found: 
    
            # sort the population in increasing order of fitness score 
            population = sorted(population, key = lambda x:x.fitness) 
            print("Generation: " + str(generation) + "\tString: " + ",".join(map(str, population[0].chromosome)) + "\tFitness: " + str(population[0].fitness)) 

            if population[0].fitness < min_fitness:
                min_fitness = population[0].fitness
                min_generation_num = generation 
                chromosome = deepcopy(population[0].chromosome)

            if (generation - min_generation_num) > stop_size / 2 and (generation - min_generation_num) < stop_size * 2 / 3: # there is no improvement
                muta_prob1 = muta_prob * 3
                print("Mutation is bigger") 
            else:    
                muta_prob1 = muta_prob

            if generation - min_generation_num > stop_size: # not updated                
                found = True 
                break
    
            # if the individual having lowest fitness score ie.  
            # 0 then we know that we have reached to the target 
            # and break the loop 
            # if population[0].fitness <= 0: 
            #     found = True
            #     break
    
            # Otherwise generate new offsprings for new generation 
            new_generation = [] 
    
            # Perform Elitism, that mean 10% of fittest population 
            # goes to the next generation 
            s = int(POPULATION_SIZE * select_prob) 
            new_generation.extend(population[:s]) 
    
            # Cross Over
            s = POPULATION_SIZE - s
            fitness_sum = -sum(row.fitness for row in population)
            
            for _ in range(s): 
                # roulette wheel selection
                # select parent1
                pick = random.uniform(0, fitness_sum)
                current = 0
                for row in population:
                    current -= row.fitness
                    if current > pick:
                        parent1 = row   
                        break;     

                # select parent1
                pick = random.uniform(0, fitness_sum)
                current = 0
                for row in population:
                    current -= row.fitness
                    if current > pick:
                        parent2 = row                
                        break;
                        
                child = parent1.crossover(parent2) 

                new_generation.append(child) 
                
            # Mutation
            for i in range(len(new_generation)): 
                rand = random.random()
                if rand < muta_prob1:
                    new_generation[i]._mutate()

            population = new_generation 
    
            generation += 1
    
        
        print("Generation: " + str(generation) + "\tString: " + "".join(map(str, chromosome)) + "\tFitness: " + str(min_fitness))         

        num_squares = len(squares)
        for r in range(num_squares):
            square = squares[r]
            height = square.height
            width = square.width
            cost = square.cost

            if chromosome[r] < 0 or chromosome[r + num_squares] < 0: 
                row = -100; column = -100
            else:
                row = chromosome[r]; column = chromosome[r + num_squares]

            squares[r] = Square(height, width, cost, row, column)     

        self.obj_val = -min_fitness;        
        self.squares = squares

        return squares
    
    
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

        self.num_instance = 0
        self.chk_cost_random = IntVar()
        self.cost_array = []

        self.status = False
        
        self.initUI()

    def initUI(self):
        top = self.top
        top.geometry("280x360+600+400")
        
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
        
        lblNumSquares = Label(frmNumSquares, text = "Number of rectangles: ")
        lblNumSquares.pack(side=LEFT)
        
        self.txtNumSquares = Entry(frmNumSquares)
        self.txtNumSquares.pack(side=RIGHT)

        frmSizeSquares = Frame(frmMain, pady = 5)
        frmSizeSquares.pack(fill=X)

        lblSizeSquares = Label(frmSizeSquares, text = "Size of rectangles: ")
        lblSizeSquares.pack(side=LEFT)

        self.txtSizeSquares = Entry(frmSizeSquares)
        self.txtSizeSquares.pack(side=RIGHT)

        frmChkRandom = Frame(frmMain, pady = 5)
        frmChkRandom.pack(fill=X)

        self.chkRandom = Checkbutton(frmChkRandom, text="Random Squares Size", variable=self.chk_random,command=self.onCheckRandom)
        self.chkRandom.pack(side=LEFT)

        frmChkCostRandom = Frame(frmMain, pady = 5)
        frmChkCostRandom.pack(fill=X)

        self.chkCostRandom = Checkbutton(frmChkCostRandom, text="Random Cost", variable=self.chk_cost_random,command=self.onCheckRandomCost)
        self.chkCostRandom.pack(side=LEFT)

        self.txtPredefinedCost = Entry(frmChkCostRandom)
        self.txtPredefinedCost.pack(side=RIGHT)

        # Multiple Instance        
        frmNumInstance = Frame(frmMain, pady = 5)
        frmNumInstance.pack(fill=X)

        lblNumInstance = Label(frmNumInstance, text = "Count of Instance: ")
        lblNumInstance.pack(side=LEFT)

        num_instance = tk.StringVar()
        num_instance.set(self.num_instance)
        self.txtNumInstance = Entry(frmNumInstance, textvariable=num_instance)
        self.txtNumInstance.pack(side=RIGHT)
        
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

    def onCheckRandomCost(self):
        if self.chk_cost_random.get():
            self.txtPredefinedCost.config(state='disabled') 
        else:    
            self.txtPredefinedCost.config(state='normal')         
        
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

        try:
            self.cost_array = []
            if self.chk_cost_random.get():                
                self.cost_array = []
            else:    
                str_cost = self.txtPredefinedCost.get()
                elements = str_cost.split(",")
                if len(elements) <= 0:
                    messagebox.showerror("Invalid File Format", "Please input predefined cost.")
                    return

                if len(elements) == self.num_squares:  # fixed cost
                    for i in range(self.num_squares):
                        self.cost_array.append(int(elements[i]))
                else:   # a predefined cost
                    for i in range(self.num_squares):
                        self.cost_array.append(int(elements[0]))
        except:
            valid = False       
               

        try:
            self.num_instance = int(self.txtNumInstance.get())
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

class RectangleEditWindow:
    def __init__(self, parent, width, height, cost):
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()

        self.width = width
        self.height = height
        self.cost = cost
        
        self.status = False
        
        self.initUI()

    def initUI(self):
        top = self.top
        top.geometry("280x200+600+400")
        
        frmMain = Frame(top, padx = 10, pady = 5)
        frmMain.pack()

        # height 
        frmHeight = Frame(frmMain, pady = 5)
        frmHeight.pack(fill=X)

        lblHeightValue = Label(frmHeight, text = "Height:  ")
        lblHeightValue.pack(side=LEFT)
        
        height_text = tk.StringVar()
        height_text.set(self.height)
        self.txtHeight = Entry(frmHeight, textvariable=height_text)
        self.txtHeight.pack(side=RIGHT)
        self.txtHeight.focus()

        # width
        frmWidth = Frame(frmMain, pady = 5)
        frmWidth.pack(fill=X)

        lblWidthValue = Label(frmWidth, text = "Width:  "  )
        lblWidthValue.pack(side=LEFT)
        
        width_text = tk.StringVar()
        width_text.set(self.width)
        self.txtWidth = Entry(frmWidth, textvariable=width_text)
        self.txtWidth.pack(side=RIGHT)

        # cost 
        frmCost = Frame(frmMain, pady = 5)
        frmCost.pack(fill=X)

        lblCostValue = Label(frmCost, text = "Cost:  ")
        lblCostValue.pack(side=LEFT)
        
        cost_text = tk.StringVar()
        cost_text.set(self.cost)
        self.txtCost = Entry(frmCost, textvariable=cost_text)
        self.txtCost.pack(side=RIGHT)
                
        frmButtons = Frame(frmMain, pady = 5)
        frmButtons.pack(fill = X, expand=True)

        btnOK = Button(frmButtons, text = "OK", width = 10, command = self.ok)
        btnOK.pack(side=LEFT, expand=True)
        
        btnCancel = Button(frmButtons, text = "Cancel", width = 10, command = self.cancel)
        btnCancel.pack(side=RIGHT, expand=True)
        
    def ok(self):
        valid = True
        
        try:
            self.width = int(self.txtWidth.get())
        except:
            valid = False

        try:
            self.height = int(self.txtHeight.get())
        except:
            valid = False

        try:
            self.cost = int(self.txtCost.get())
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

        self.tblSquare.bind("<Double-Button-1>", self.on_double_click_table)
        
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

        self.cb_method = ttk.Combobox(frmControl, values=("MIP", 
                                                    "Brute Force", 
                                                    "Greedy Arbitrary", 
                                                    "Greedy Decreaing Width", 
                                                    "Greedy Decreaing Height", 
                                                    "Greedy Decreaing Area", 
                                                    "Greedy Decreaing Cost", 
                                                    "Greedy Decreaing Area * Cost",
                                                    "GA",                                                    
                                                    ))
        self.cb_method.bind('<<ComboboxSelected>>', self.on_change_method)    
        self.cb_method.set("GA")
        self.cb_method.pack(side=LEFT)

        self.chk_local_search = IntVar()
        
        self.chkLocalSearch = Checkbutton(frmControl, text="Local Search", variable=self.chk_local_search)
        self.chkLocalSearch.pack(fill=Y, expand=False, side=LEFT)

        
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

        frmBoardSpace6 = Frame(frmControl, width=14)
        frmBoardSpace6.pack(fill=Y, expand=False, side=RIGHT)
        
        self.btnBoard = Button(frmControl, text="Refresh Board", width=22, command=self.refresh_board)
        self.btnBoard.pack(fill=Y, expand=False, side=RIGHT)
        
        frmBoardSpace3 = Frame(self, height=2)
        frmBoardSpace3.pack(fill=BOTH)

        frmControl1 = Frame(self)
        frmControl1.pack(fill=BOTH, expand=False)

          
        lblPopulationSize = Label(frmControl1, text = "Population Size: ")
        lblPopulationSize.pack(side=LEFT)
        
        pop_size = tk.StringVar()
        pop_size.set(100)
        self.txtPopulationSize = Entry(frmControl1, textvariable=pop_size, width=10)
        self.txtPopulationSize.pack(side=LEFT)

        lblStopSize = Label(frmControl1, text = "Stop Size: ")
        lblStopSize.pack(fill=Y, expand=False, side=LEFT)
        
        stop_size = tk.StringVar()
        stop_size.set(200)
        self.txtStopSize = Entry(frmControl1, textvariable=stop_size, width=10)
        self.txtStopSize.pack(fill=Y, expand=False, side=LEFT)

        lblMutaProb = Label(frmControl1, text = "Muation Probablity: ")
        lblMutaProb.pack(fill=Y, expand=False, side=LEFT)
        
        muta_prob = tk.StringVar()
        muta_prob.set(0.1)
        self.txtMutaProb = Entry(frmControl1, textvariable=muta_prob, width=10)
        self.txtMutaProb.pack(fill=Y, expand=False, side=LEFT)

        lblSelectProb = Label(frmControl1, text = "Selection Percent: ")
        lblSelectProb.pack(fill=Y, expand=False, side=LEFT)
        
        select_prob = tk.StringVar()
        select_prob.set(0.1)
        self.txtSelectProb = Entry(frmControl1, textvariable=select_prob, width=10)
        self.txtSelectProb.pack(fill=Y, expand=False, side=LEFT)

        self.btnBatchSolve = Button(frmControl1, text="Solve Batch", width=22, command=self.solve_batch_problem)
        self.btnBatchSolve.pack(fill=Y, expand=False, side=RIGHT)

        self.on_change_method(1)
        
    def import_problem(self):
        filename = filedialog.askopenfilename(title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
        if not filename:
            return
        self.problem.load(filename)
        self.cnsBoard.delete("all")

        if self.problem.obj_val > -1000000:
            self.display_problem(solution=True)
        else:            
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
        chk_cost_random = problem_dialog.chk_cost_random.get()
        cost_array = problem_dialog.cost_array

        self.problem.generate(class_num, num_rows, num_columns, num_squares, size_squares, chk_random, chk_cost_random, cost_array)
        self.cnsBoard.delete("all")
        self.display_problem()

        # generate multiple instance
        if problem_dialog.num_instance < 1:
            return

        # generate directory name
        rect_size_str = ""
        if chk_random:
            rect_size_str = "rnd"
        else:    
            rect_size_str = "{}x{}".format(size_squares, size_squares)

        cost_str = ""
        if chk_cost_random:
            cost_str = "rnd"
        else:    
            cost_str = "fixed"
        dir_path = "{}/IC{}_{}x{}_{}R_S{}_cost_{}".format(os.getcwd(), class_num, num_rows, num_columns, num_squares, rect_size_str, cost_str)    
        
        try:
            os.mkdir(dir_path)
        except OSError:
            print ("Creation of the directory %s failed" % dir_path)
        else:
            print ("Successfully created the directory %s " % dir_path)

        for i in range(problem_dialog.num_instance):
            instance_name = "{}/board_instance_{}.txt".format(dir_path, i + 1)
            self.problem.refreshValues()
            self.problem.save(instance_name)


    def refresh_board(self):            
        self.problem.refreshValues()
        self.cnsBoard.delete("all")
        self.display_problem()
    
    def solve_problem(self):
        overlap = self.chk_overlap.get()
        local_search = self.chk_local_search.get()
        start = time.time()

        method = self.cb_method.get()
        if method == "MIP": 
            self.problem.solve(overlap)

        if method == "Brute Force" :
            self.problem.solve_by_brute_force(overlap)

        if method == "Greedy Arbitrary" :
            self.problem.solve_by_greedy(overlap, "random", local_search)    
            
        if method == "Greedy Decreaing Width" :
            self.problem.solve_by_greedy(overlap, "width", local_search)    

        if method == "Greedy Decreaing Height" :
            self.problem.solve_by_greedy(overlap, "height", local_search)    

        if method == "Greedy Decreaing Area" :
            self.problem.solve_by_greedy(overlap, "area", local_search)    

        if method == "Greedy Decreaing Cost" :
            self.problem.solve_by_greedy(overlap, "cost", local_search)    

        if method == "Greedy Decreaing Area * Cost" :
            self.problem.solve_by_greedy(overlap, "area_cost", local_search)    

        if method == "GA":
            popu_size = int(self.txtPopulationSize.get())
            stop_size = int(self.txtStopSize.get())
            muta_prob = float(self.txtMutaProb.get())
            select_prob = float(self.txtSelectProb.get())
            self.problem.solve_by_ga(overlap, popu_size, stop_size, muta_prob, select_prob)        
            
        end = time.time()
        self.display_problem(solution=True)

        gap = (end - start)
        elapsed = time.strftime('%H:%M:%S', time.gmtime(gap))
        self.master.title("Board Packing Problem" + " ------ Running Time = " + elapsed)
        
    def solve_batch_problem(self):
        dir_path = filedialog.askdirectory(title = "Select Directory")
        if not dir_path:
            return     

        print(dir_path)       
        onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

        total_count = len(onlyfiles)
        if total_count < 1:
            messagebox.showerror("Invalid Directory", "There is no file in this directory")
            return;

        start = time.time()

        method = self.cb_method.get()
        output_dir = "{}/{}".format(dir_path, method)

        try:
            os.mkdir(output_dir)
        except OSError:
            print ("Creation of the directory %s failed" % output_dir)
        else:
            print ("Successfully created the directory %s " % output_dir)

        
        for i in range(total_count):
            path = "{}/{}".format(dir_path, onlyfiles[i])
            self.problem.load(path)
            self.solve_problem()
            save_path = "{}/{}".format(output_dir, onlyfiles[i])
            self.problem.save(save_path)

        end = time.time()

        gap = (end - start) / total_count
        elapsed = time.strftime('%H:%M:%S', time.gmtime(gap))
        self.master.title("Board Packing Problem: Instance Count = " + str(total_count)  + " ------ Average Running Time = " + elapsed)

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
                h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
                r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]

                fill = '#%02x%02x%02x' % (r, g, b)

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
                if square.row >= 0:
                    tags = (str(key + 1),)
                else:    
                    tags = ""
            else:                
                if key % 2 == 0:
                    tags = ("white",)
                else:
                    tags = ("grey",)                    

            table.insert("", "end", None, text=str(key + 1), values=(str(square.height), str(square.width), str(square.row + 1), str(square.column + 1), str(square.cost), str(value), str(profit)), tags=tags)

            total_count += 1

        table.insert("", "end", None, text=str(total_count + 1), values=("Total", "", "", "", str(total_cost), str(total_value), str(self.problem.obj_val)), tags=("green",))
        
        
                
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

    def on_double_click_table(self, event):
        item_id = event.widget.focus()
        item = event.widget.item(item_id)

        num = int(item['text']);
        values = item['values']        

        squares = self.problem.squares
        if num > len(squares):
            return;
        
        # display rectangle edit dialog
        dialog = RectangleEditWindow(root, values[0], values[1], values[4])
        
        root.wait_window(dialog.top)
        
        if not dialog.status:
            return
        
        width = dialog.width
        height = dialog.height
        cost = dialog.cost

        square = Square(height, width, cost, -1, -1)
        squares[num - 1] = square
        
        print(item_id, width, height, cost)

        self.display_problem()

    def on_change_method(self, event):
        method = self.cb_method.get()
        if method == "MIP": 
            self.chkLocalSearch.config(state='disabled') 

        if method == "Brute Force" :
            self.chkLocalSearch.config(state='disabled')

        if method == "Greedy Arbitrary" :
            self.chkLocalSearch.config(state='normal')
            
        if method == "Greedy Decreaing Width" :
            self.chkLocalSearch.config(state='normal')

        if method == "Greedy Decreaing Height" :
            self.chkLocalSearch.config(state='normal')

        if method == "Greedy Decreaing Area" :
            self.chkLocalSearch.config(state='normal')

        if method == "Greedy Decreaing Cost" :
            self.chkLocalSearch.config(state='normal')

        if method == "Greedy Decreaing Area * Cost" :
            self.chkLocalSearch.config(state='normal')

        if method == "GA":
            self.chkLocalSearch.config(state='disabled') 
                
app = MainWindow()

root.mainloop()