// Data
int nRows = ...;
int nCols = ...;

range rows = 1..nRows;
range cols = 1..nCols;
int g[rows][cols] = ...;

int nRectangle = 3;

tuple Rectangle {
   int w;
   int h;
   int c;
}; 

Rectangle r = ...;

// Variables