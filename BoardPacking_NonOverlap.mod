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


execute
{
  	for(r in K)
  	{
 		writeln(rect_array[r]);
 	} 		
}

