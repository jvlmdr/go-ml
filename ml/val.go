package ml

// Perf describes the performance of a binary classification test.
type Perf struct {
	TP, TN, FP, FN int
}

// Prec returns the precision of the classifier.
func (r Perf) Prec() float64 {
	// True positives over reported positives.
	return float64(r.TP) / float64(r.TP+r.FP)
}

// Recall returns the recall of the classifier.
func (r Perf) Recall() float64 {
	// True positives over actual positives.
	return float64(r.TP) / float64(r.TP+r.FN)
}

// Acc returns the accuracy of the classifications.
func (r Perf) Acc() float64 {
	// Fraction correct.
	return float64(r.TP+r.TN) / float64(r.TP+r.TN+r.FP+r.FN)
}

// F1 returns the F1 score.
func (r Perf) F1() float64 {
	return 2 * float64(r.TP) / float64(2*r.TP+r.FP+r.FN)
}

// TPR computes True Positive Rate.
// Equal to precision.
func (r Perf) TPR() float64 {
	return float64(r.TP) / float64(r.TP+r.FN)
}

// TNR computes True Negative Rate.
func (r Perf) TNR() float64 {
	return float64(r.TN) / float64(r.TN+r.FP)
}

// FPR computes False Positive Rate.
func (r Perf) FPR() float64 {
	return float64(r.FP) / float64(r.FP+r.TN)
}

// FNR computes False Negative Rate.
// Equal to one minus recall.
func (r Perf) FNR() float64 {
	return float64(r.FN) / float64(r.FN+r.TP)
}
