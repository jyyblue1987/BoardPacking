// Data
int nRows = ...;
int nCols = ...;

range rows = 1..nRows;
range cols = 1..nCols;
int g[rows][cols] = ...;

tuple Rectangle {
   int w;
   int h;
   int c;
};

int nRectangle = ...;
range K = 1..nRectangle;
 
Rectangle rect_array[K] = ...;

// Variables
dvar boolean x[K][rows][cols];
dvar boolean y[rows][cols];

// macro
dexpr int TotalCost = sum(i in rows, j in cols)g[i][j]*y[i][j] - sum(r in K, i in rows, j in cols)rect_array[r].c * x[r][i][j];

// Objective 
maximize
  	TotalCost;

// contraints
subject to {
  	forall( r in K )
	    eachRectangle:
	      	sum( i in rows, j in cols ) 
	        	x[r][i][j] <= 1;
	        	
	forall( i in rows, j in cols : g[i][j] > 0 )
	  	g_bigger:
	  		sum(r in K, u in rows, v in cols: u <= i && i < u + rect_array[r].h && v <= j && j < v + rect_array[r].w  ) // (u, v) - B(i,j,k)
	  		  	x[r][u][v] >= y[i][j];
	  		  	
	forall( i in rows, j in cols : g[i][j] < 0 )
	  	g_smaller:
	  		sum(r in K, u in rows, v in cols: u <= i && i < u + rect_array[r].h && v <= j && j < v + rect_array[r].w  ) // (u, v) - B(i,j,k)
	  		  	x[r][u][v] <= nRectangle * y[i][j];  		  	
	  		  	
	forall( r in K, i in rows, j in cols: i + rect_array[r].h > nRows + 1 || j + rect_array[r].w > nCols + 1 )
	  	trivialVar:
	  		x[r][i][j] == 0;
}	  		
