package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/jackvalmadre/go-ml"
	"log"
	"math"
	"os"
	"strconv"
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "%s pos-scores.txt neg-scores.txt roc.txt\n", os.Args[0])
	flag.PrintDefaults()
}

func main() {
	log.SetOutput(os.Stdout)
	flag.Usage = usage
	flag.Parse()

	if flag.NArg() != 3 {
		flag.Usage()
		os.Exit(1)
	}
	posFilename := flag.Arg(0)
	negFilename := flag.Arg(1)
	rocFilename := flag.Arg(2)

	// Read scores.
	pos, err := readScores(posFilename)
	if err != nil {
		log.Fatalln(err)
	}
	neg, err := readScores(negFilename)
	if err != nil {
		log.Fatalln(err)
	}

	results := ml.EnumerateResults(pos, neg)

	// Open ROC file for writing.
	file, err := os.Create(rocFilename)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()
	fmt.Fprintln(file, "Recall\tPrecision")
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
		fmt.Fprintf(file, "%g\t%g\n", r, p)
		prev = r
	}
	fmt.Println("Average Precision:", ap)
}

func readScores(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// Read lines.
	var scores []float64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		score, err := strconv.ParseFloat(scanner.Text(), 64)
		if err != nil {
			return nil, err
		}
		scores = append(scores, score)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return scores, nil
}
