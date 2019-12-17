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

tuple Position {
   int i;
   int j;   
};


int nRectangle = ...;
range K = 1..nRectangle;
 
Rectangle rect_array[K] = ...;

// definition of A(k, i, j)
//Position A[K] = ...;

// profit
int p[K][rows][cols];

execute Profit_Initialize
{
  	for(r in K)
  	{
  	  	var w = rect_array[r].w;
  	  	var h = rect_array[r].h;
  	  	var c = rect_array[r].c;
  	  	writeln(w, h, c);
  	  	
  	  	for(i in rows)
  	  	{
  	  	  	for(j in cols)
  	  	  	{
  	  	  	  	if( i + h > nRows + 1 || j + w > nCols + 1 )
  	  				p[r][i][j] = 0;
  	  			else
  	  			{
  	  			  	var sum = 0;
  	  			  	var end_row = i + h;
  	  			  	var end_col = j + w;
  	  			  	for(var ii = i; ii < end_row; ii++)
  	  			  	{
  	  			  	  	for(var jj = j; jj < end_col; jj++)
  	  			  	  	{
  	  			  	  	  	sum += g[ii][jj];
  	  			  	  	}
  	  			  	}
  	  			  	
  	  			  	p[r][i][j] = sum - c;
  	  			}	  	  
  	  			
  	  			writeln(r, '-', i, '-', j, '-', p[r][i][j]);		
  	  	  	}
  	  	}
 		
 	} 		
}


// Variables
dvar boolean x[K][rows][cols];

// Objective 
//minimize
//  sum(k in K, i in rows, j in cols)
//    p[k][i][j] * x[k][i][j];


// contraints
subject to {
  	forall( r in K )
	    eachRectangle:
	      	sum( i in rows, j in cols ) 
	        	x[r][i][j] <= 1;
	        	
	forall( i in rows, j in cols )
	  	nonOverlap:
	  		sum(r in K, u in rows, v in cols: u <= i && i < u + rect_array[r].h && v <= j && j < v + rect_array[r].w  ) // (u, v) - B(i,j,k)
	  		  	x[r][u][v] <= 1;
	  		  	
	forall( r in K, i in rows, j in cols: i + rect_array[r].h > nRows + 1 || j + rect_array[r].w > nCols + 1 )
	  	inFeasible:
	  		x[r][i][j] == 0;
}	  		
