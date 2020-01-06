from docplex.mp.model import Model
from docplex.util.environment import get_environment

def build_packing_problem():
    mdl = Model('packing')

    
    nCols = 10
    nRectangle = 3

    rect_array = [
        (5,5,3),
        (6,5,2),
        (3,6,2)
    ];

    K = range(nRectangle)

    g = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,0],
        [0,0,0,0,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]
    ];

    nRows = len(g)
    nCols = len(g[0])

    rows = range(nRows)
    cols = range(nCols)
    
    p = [[[0 for j in cols] for i in rows] for r in K]
    
    # calculate profit
    for r in K:
        w = rect_array[r][0]
        h = rect_array[r][1]
        c = rect_array[r][2]

        for i in rows :
            for j in cols :
                if i + h <= nRows and j + w <= nCols :
                    sum = 0
                    end_row = i + h
                    end_col = j + w
                    
                    for ii in range(i, end_row - 1):
                        for jj in range(j, end_col - 1):
                            sum += g[ii][jj]

                    p[r][i][j] = sum - c        


    print(p)

    return mdl;

def print_solution(mdl):
    obj = mdl.objective_value
    print("* Production model solved with objective: {:g}".format(obj))
    

# ----------------------------------------------------------------------------
# Solve the model and display the result
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Build the model
    model = build_packing_problem()
    model.print_information()
    # Solve the model.
    if model.solve():
        print_solution(model)
        # Save the CPLEX solution as "solution.json" program output
        with get_environment().get_output_stream("solution.json") as fp:
            model.solution.export(fp, "json")
    else:
        print("Problem has no solution")    