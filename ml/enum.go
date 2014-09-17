package ml

import "math"

// PerfPath describes the performance along a path of operating points.
// Ordered from most negative (TN+FN) classifications to most positive (TP+FP).
type PerfPath []Perf

// AvgPrec computes the average precision of a range of operating points.
func (perfs PerfPath) AvgPrec() float64 {
	var ap float64
	var prev float64
	for _, x := range perfs {
		p := x.Prec()
		if math.IsNaN(p) {
			continue
		}
		r := x.Recall()
		dr := r - prev
		ap += p * dr
		prev = r
	}
	return ap
}
