package ml

import "sort"

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

// Associates a labelled example with a score.
type Example struct {
	Positive bool
	Score    float64
}

func EnumerateResults(pos, neg []float64) []Result {
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

type ByScore []Example
func (s ByScore) Len() int { return len(s) }
func (s ByScore) Less(i, j int) bool { return s[i].Score < s[j].Score }
func (s ByScore) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
