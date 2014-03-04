package ml

import (
	"math"
	"sort"
)

// Binary classification result.
type Result struct {
	TP int
	TN int
	FP int
	FN int
}

func (r Result) Precision() float64 {
	// True positives over reported positives.
	return float64(r.TP) / float64(r.TP+r.FP)
}

func (r Result) Recall() float64 {
	// True positives over actual positives.
	return float64(r.TP) / float64(r.TP+r.FN)
}

func (r Result) Accuracy() float64 {
	// Fraction correct.
	return float64(r.TP+r.TN) / float64(r.TP+r.TN+r.FP+r.FN)
}

type ResultSet []Result

// Associates a labelled example with a score.
type Example struct {
	Positive bool
	Score    float64
}

// Inputs need not be sorted.
func EnumerateResults(pos, neg []float64) ResultSet {
	// Merge scores.
	var examples []Example
	for _, score := range pos {
		examples = append(examples, Example{true, score})
	}
	for _, score := range neg {
		examples = append(examples, Example{false, score})
	}
	// Sort from highest score to lowest.
	sort.Sort(sort.Reverse(ByScore(examples)))

	var results []Result
	// Positives correctly classified as positive.
	tp := 0
	// Start with everything classified as negative (zero recall).
	for p := 0; p <= len(examples); p++ {
		if p > 0 {
			// As threshold is lowered, positives occur.
			// Check if new positive is correctly classified.
			if examples[p-1].Positive {
				tp++
			}
		}
		// Negatives incorrectly classified as positive.
		fp := p - tp
		// Positives incorrectly classified as negative.
		fn := len(pos) - tp
		// Negatives correctly classified as negative.
		tn := len(neg) - fp
		results = append(results, Result{tp, tn, fp, fn})
	}
	return results
}

func (results ResultSet) AveragePrecision() float64 {
	var ap float64
	var prev float64
	for _, c := range results {
		p := c.Precision()
		if math.IsNaN(p) {
			continue
		}
		r := c.Recall()
		dr := r - prev
		ap += p * dr
		prev = r
	}
	return ap
}

type ByScore []Example
func (s ByScore) Len() int { return len(s) }
func (s ByScore) Less(i, j int) bool { return s[i].Score < s[j].Score }
func (s ByScore) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
