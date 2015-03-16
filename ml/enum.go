package ml

import (
	"math"
	"sort"
)

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

// ValScore is a confidence score of an example whose
// ground truth class (positive or negative) is known.
// Higher scores correspond to the positive class.
type ValScore struct {
	Score float64
	Pos   bool
}

// Enum computes a performance range from a sorted list
// of scores of examples with ground truth labels.
func Enum(vals []ValScore) PerfPath {
	if !sort.IsSorted(byScoreDesc(vals)) {
		panic("not sorted")
	}
	// Count number of positive and negative.
	var pos, neg int
	for _, val := range vals {
		if val.Pos {
			pos++
		} else {
			neg++
		}
	}
	// Start with high threshold, everything negative,
	// then gradually lower it.
	perfs := make([]Perf, 0, len(vals)+1)
	perf := Perf{FN: pos, TN: neg}
	perfs = append(perfs, perf)
	for _, val := range vals {
		if val.Pos {
			// Positive example classified as positive instead of negative.
			perf.TP, perf.FN = perf.TP+1, perf.FN-1
		} else {
			// Negative example classified as positive instead of negative.
			perf.FP, perf.TN = perf.FP+1, perf.TN-1
		}
		perfs = append(perfs, perf)
	}
	return PerfPath(perfs)
}

type byScoreDesc []ValScore

func (s byScoreDesc) Len() int           { return len(s) }
func (s byScoreDesc) Less(i, j int) bool { return s[i].Score > s[j].Score }
func (s byScoreDesc) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func Sort(vals []ValScore) {
	sort.Sort(byScoreDesc(vals))
}
