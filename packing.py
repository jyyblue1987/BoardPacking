from docplex.mp.model import Model
from docplex.util.environment import get_environment

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
        w = rect_array[r][0]
        h = rect_array[r][1]
        c = rect_array[r][2]

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


    print(p)

    # declar var
    idx = [(r, i, j) for r in K for i in rows for j in cols ]
    mdl.x = mdl.binary_var_dict(idx, None, None, "X")

   
    # contraints

    # make sure that only one location is chosen for each rectangle
    mdl.add_constraints(mdl.sum(mdl.x[r, i, j] for i in rows for j in cols) <= 1 for r in K)

    # make sure that only one location is chosen for each rectangle
    mdl.add_constraints(mdl.sum(mdl.x[r, u, v] for r in K for u in rows for v in cols 
                        if u <= i and i < u + rect_array[r][1] and v <= j and j < v + rect_array[r][0] ) <= 1 
                        for i in rows for j in cols)

    mdl.add_constraints(mdl.x[r, i, j] == 0 for r in K for i in rows for j in cols if i + rect_array[r][1] > nRows or j + rect_array[r][0] > nCols )

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
                        if u <= i and i < u + rect_array[r][1] and v <= j and j < v + rect_array[r][0] ) >= mdl.y[i, j] 
                        for i in rows for j in cols if g[i][j] > 0)

    # smaller
    mdl.add_constraints(mdl.sum(mdl.x[r, u, v] for r in K for u in rows for v in cols 
                        if u <= i and i < u + rect_array[r][1] and v <= j and j < v + rect_array[r][0] ) <= nRectangle * mdl.y[i, j] 
                        for i in rows for j in cols if g[i][j] < 0)   

    mdl.add_constraints(mdl.x[r, i, j] == 0 for r in K for i in rows for j in cols if i + rect_array[r][1] > nRows or j + rect_array[r][0] > nCols )

    # objective values
    mdl.maximize(mdl.sum(mdl.y[i, j] * g[i][j] for i in rows for j in cols) - mdl.sum(mdl.x[r, i, j] * rect_array[r][2] for r in K for i in rows for j in cols))
    
    return mdl;    

def print_solution(mdl, rect_array, g):
    obj = mdl.objective_value

    nRectangle = len(rect_array)
    K = range(nRectangle)

    nRows = len(g)
    nCols = len(g[0])

    rows = range(nRows)
    cols = range(nCols)

    print("* Production model solved with objective: {:g}".format(obj))
    for r in K:
        w = rect_array[r][0]
        h = rect_array[r][1]

        left = 0; top = 0

        for i in rows :
            for j in cols:
                if mdl.x[r, i, j].solution_value > 0 :
                    left = j + 1
                    top = i + 1
                    print(top, left, w, h)


# ----------------------------------------------------------------------------
# Solve the model and display the result
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Build the model
    rect_array = [
        (5,5,3),
        (5,6,2),
        (6,3,2)
    ];

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

    model = build_packing_nonoverlay_problem(rect_array, g)
    
    # Solve the model.
    if model.solve():
        print_solution(model, rect_array, g)
        # Save the CPLEX solution as "solution.json" program output
        with get_environment().get_output_stream("solution_nonoverlay.json") as fp:
            model.solution.export(fp, "json")
    else:
        print("Problem has no solution")    


    model = build_packing_overlay_problem(rect_array, g)
    
    # Solve the model.
    if model.solve():
        print_solution(model, rect_array, g)
        # Save the CPLEX solution as "solution.json" program output
        with get_environment().get_output_stream("solution_overlay.json") as fp:
            model.solution.export(fp, "json")
    else:
        print("Problem has no solution")        


