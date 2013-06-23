set term postscript eps color
set output "roc.eps"

set nokey
set ylabel "Precision"
set xlabel "Recall"
set xrange [0:1]
set yrange [0:1]
set size square
set xtics (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
set ytics (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
set grid

plot "roc.txt" using 1:2 w l
