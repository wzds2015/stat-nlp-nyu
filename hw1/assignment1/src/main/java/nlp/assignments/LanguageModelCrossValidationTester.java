package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.text.NumberFormat;
import java.text.DecimalFormat;

import nlp.langmodel.LanguageModel;
import nlp.util.CommandLineUtils;

/**
 * This is the main harness for assignment 1. To run this harness, use
 * <p/>
 * java nlp.assignments.LanguageModelTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system. Second, find the point
 * in the main method (near the bottom) where an EmpiricalUnigramLanguageModel
 * is constructed. You will be writing new implementations of the LanguageModel
 * interface and constructing them there.
 */
public class LanguageModelCrossValidationTester {

    // HELPER CLASS FOR THE HARNESS, CAN IGNORE
    static class EditDistance {
        static double INSERT_COST = 1.0;
        static double DELETE_COST = 1.0;
        static double SUBSTITUTE_COST = 1.0;

        private double[][] initialize(double[][] d) {
            for (int i = 0; i < d.length; i++) {
                for (int j = 0; j < d[i].length; j++) {
                    d[i][j] = Double.NaN;
                }
            }
            return d;
        }

        public double getDistance(List<? extends Object> firstList,
                                  List<? extends Object> secondList) {
            double[][] bestDistances = initialize(new double[firstList.size() + 1][secondList
                    .size() + 1]);
            return getDistance(firstList, secondList, 0, 0, bestDistances);
        }

        private double getDistance(List<? extends Object> firstList,
                                   List<? extends Object> secondList, int firstPosition,
                                   int secondPosition, double[][] bestDistances) {
            if (firstPosition > firstList.size()
                    || secondPosition > secondList.size())
                return Double.POSITIVE_INFINITY;
            if (firstPosition == firstList.size()
                    && secondPosition == secondList.size())
                return 0.0;
            if (Double.isNaN(bestDistances[firstPosition][secondPosition])) {
                double distance = Double.POSITIVE_INFINITY;
                distance = Math.min(
                        distance,
                        INSERT_COST
                                + getDistance(firstList, secondList,
                                firstPosition + 1, secondPosition,
                                bestDistances));
                distance = Math.min(
                        distance,
                        DELETE_COST
                                + getDistance(firstList, secondList,
                                firstPosition, secondPosition + 1,
                                bestDistances));
                distance = Math.min(
                        distance,
                        SUBSTITUTE_COST
                                + getDistance(firstList, secondList,
                                firstPosition + 1, secondPosition + 1,
                                bestDistances));
                if (firstPosition < firstList.size()
                        && secondPosition < secondList.size()) {
                    if (firstList.get(firstPosition).equals(
                            secondList.get(secondPosition))) {
                        distance = Math.min(
                                distance,
                                getDistance(firstList, secondList,
                                        firstPosition + 1, secondPosition + 1,
                                        bestDistances));
                    }
                }
                bestDistances[firstPosition][secondPosition] = distance;
            }
            return bestDistances[firstPosition][secondPosition];
        }
    }

    // HELPER CLASS FOR THE HARNESS, CAN IGNORE
    static class SentenceCollection extends AbstractCollection<List<String>> {
        static class SentenceIterator implements Iterator<List<String>> {

            BufferedReader reader;

            public boolean hasNext() {
                try {
                    return reader.ready();
                } catch (IOException e) {
                    return false;
                }
            }

            public List<String> next() {
                try {
                    String line = reader.readLine();
                    String[] words = line.split("\\s+");
                    List<String> sentence = new ArrayList<String>();
                    for (int i = 0; i < words.length; i++) {
                        String word = words[i];
                        sentence.add(word.toLowerCase());
                    }
                    return sentence;
                } catch (IOException e) {
                    throw new NoSuchElementException();
                }
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }

            public SentenceIterator(BufferedReader reader) {
                this.reader = reader;
            }
        }

        String fileName;

        public Iterator<List<String>> iterator() {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(
                        fileName));
                return new SentenceIterator(reader);
            } catch (FileNotFoundException e) {
                throw new RuntimeException("Problem with SentenceIterator for "
                        + fileName);
            }
        }

        public int size() {
            int size = 0;
            Iterator<List<String>> i = iterator();
            while (i.hasNext()) {
                size++;
                i.next();
            }
            return size;
        }

        public SentenceCollection(String fileName) {
            this.fileName = fileName;
        }

        public static class Reader {
            static Collection<List<String>> readSentenceCollection(
                    String fileName) {
                return new SentenceCollection(fileName);
            }
        }

    }

    static double calculatePerplexity(LanguageModel languageModel,
                                      Collection<List<String>> sentenceCollection) {
        double logProbability = 0.0;
        double numSymbols = 0.0;
        for (List<String> sentence : sentenceCollection) {
            logProbability += Math.log(languageModel
                    .getSentenceProbability(sentence)) / Math.log(2.0);
            numSymbols += sentence.size();
        }
        double avgLogProbability = logProbability / numSymbols;
        // 2^(-perp)
        double perplexity = Math.pow(0.5, avgLogProbability);
        return perplexity;
    }

    static double calculateWordErrorRate(LanguageModel languageModel,
                                         List<SpeechNBestList> speechNBestLists, boolean verbose) {
        double totalDistance = 0.0;
        double totalWords = 0.0;
        EditDistance editDistance = new EditDistance();
        for (SpeechNBestList speechNBestList : speechNBestLists) {
            List<String> correctSentence = speechNBestList.getCorrectSentence();
            List<String> bestGuess = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            double numWithBestScores = 0.0;
            double distanceForBestScores = 0.0;
            for (List<String> guess : speechNBestList.getNBestSentences()) {
                double score = Math.log(languageModel
                        .getSentenceProbability(guess))
                        + (speechNBestList.getAcousticScore(guess) / 16.0);
                double distance = editDistance.getDistance(correctSentence,
                        guess);
                if (score == bestScore) {
                    numWithBestScores += 1.0;
                    distanceForBestScores += distance;
                }
                if (score > bestScore || bestGuess == null) {
                    bestScore = score;
                    bestGuess = guess;
                    distanceForBestScores = distance;
                    numWithBestScores = 1.0;
                }
            }
            // double distance = editDistance.getDistance(correctSentence,
            // bestGuess);
            totalDistance += distanceForBestScores / numWithBestScores;
            totalWords += correctSentence.size();
            if (verbose) {
                System.out.println();
                displayHypothesis("GUESS:", bestGuess, speechNBestList,
                        languageModel);
                displayHypothesis("GOLD:", correctSentence, speechNBestList,
                        languageModel);
            }
        }
        return totalDistance / totalWords;
    }

    private static NumberFormat nf = new DecimalFormat("0.00E00");

    private static void displayHypothesis(String prefix, List<String> guess,
                                          SpeechNBestList speechNBestList, LanguageModel languageModel) {
        double acoustic = speechNBestList.getAcousticScore(guess) / 16.0;
        double language = Math.log(languageModel.getSentenceProbability(guess));
        System.out.println(prefix + "\tAM: " + nf.format(acoustic) + "\tLM: "
                + nf.format(language) + "\tTotal: "
                + nf.format(acoustic + language) + "\t" + guess);
    }

    static double calculateWordErrorRateLowerBound(
            List<SpeechNBestList> speechNBestLists) {
        double totalDistance = 0.0;
        double totalWords = 0.0;
        EditDistance editDistance = new EditDistance();
        for (SpeechNBestList speechNBestList : speechNBestLists) {
            List<String> correctSentence = speechNBestList.getCorrectSentence();
            double bestDistance = Double.POSITIVE_INFINITY;
            for (List<String> guess : speechNBestList.getNBestSentences()) {
                double distance = editDistance.getDistance(correctSentence,
                        guess);
                if (distance < bestDistance)
                    bestDistance = distance;
            }
            totalDistance += bestDistance;
            totalWords += correctSentence.size();
        }
        return totalDistance / totalWords;
    }

    static double calculateWordErrorRateUpperBound(
            List<SpeechNBestList> speechNBestLists) {
        double totalDistance = 0.0;
        double totalWords = 0.0;
        EditDistance editDistance = new EditDistance();
        for (SpeechNBestList speechNBestList : speechNBestLists) {
            List<String> correctSentence = speechNBestList.getCorrectSentence();
            double worstDistance = Double.NEGATIVE_INFINITY;
            for (List<String> guess : speechNBestList.getNBestSentences()) {
                double distance = editDistance.getDistance(correctSentence,
                        guess);
                if (distance > worstDistance)
                    worstDistance = distance;
            }
            totalDistance += worstDistance;
            totalWords += correctSentence.size();
        }
        return totalDistance / totalWords;
    }

    static double calculateWordErrorRateRandomChoice(
            List<SpeechNBestList> speechNBestLists) {
        double totalDistance = 0.0;
        double totalWords = 0.0;
        EditDistance editDistance = new EditDistance();
        for (SpeechNBestList speechNBestList : speechNBestLists) {
            List<String> correctSentence = speechNBestList.getCorrectSentence();
            double sumDistance = 0.0;
            double numGuesses = 0.0;
            for (List<String> guess : speechNBestList.getNBestSentences()) {
                double distance = editDistance.getDistance(correctSentence,
                        guess);
                sumDistance += distance;
                numGuesses += 1.0;
            }
            totalDistance += sumDistance / numGuesses;
            totalWords += correctSentence.size();
        }
        return totalDistance / totalWords;
    }

    static Collection<List<String>> extractCorrectSentenceList(
            List<SpeechNBestList> speechNBestLists) {
        Collection<List<String>> correctSentences = new ArrayList<List<String>>();
        for (SpeechNBestList speechNBestList : speechNBestLists) {
            correctSentences.add(speechNBestList.getCorrectSentence());
        }
        return correctSentences;
    }

    static Set<String> extractVocabulary(
            Collection<List<String>> sentenceCollection) {
        Set<String> vocabulary = new HashSet<String>();
        for (List<String> sentence : sentenceCollection) {
            for (String word : sentence) {
                vocabulary.add(word);
            }
        }
        return vocabulary;
    }

//    static double[] retrieveParameterFromInd(int ind) {
//        int K = (ind % 11) * 5 + 10;
//        double lambda1, lambda2;
//        int lambdaInd = ind / 11;
//        if (lambdaInd <= 9) { lambda1 = 0.1; lambda2 = lambdaInd * 0.1; }
//        else if (lambdaInd > 9  && lambdaInd <= 17) { lambda1 = 0.2; lambda2 = (lambdaInd-9 )*0.1; }
//        else if (lambdaInd > 17 && lambdaInd <= 24) { lambda1 = 0.3; lambda2 = (lambdaInd-17)*0.1; }
//        else if (lambdaInd > 24 && lambdaInd <= 30) { lambda1 = 0.4; lambda2 = (lambdaInd-24)*0.1; }
//        else if (lambdaInd > 30 && lambdaInd <= 35) { lambda1 = 0.5; lambda2 = (lambdaInd-30)*0.1; }
//        else if (lambdaInd > 35 && lambdaInd <= 39) { lambda1 = 0.6; lambda2 = (lambdaInd-35)*0.1; }
//        else if (lambdaInd > 39 && lambdaInd <= 42) { lambda1 = 0.7; lambda2 = (lambdaInd-39)*0.1; }
//        else if (lambdaInd > 42 && lambdaInd <= 44) { lambda1 = 0.8; lambda2 = (lambdaInd-42)*0.1; }
//        else { lambda1 = 0.9; lambda2 = 0.1; }
//        return new double[]{lambda1, lambda2, K};
//    }

    static void crossValidationWriter(double[] l1, double[] l2, double[] k, double[] perp, int len) {

        try {
            FileWriter writer = new FileWriter("/Users/admin/Desktop/CV_NLP.txt", true);
            for (int i=0; i<len; i++) {
                if (i != len-1) { writer.write(l1[i]+","+l2[i]+","+k[i]+","+perp[i]+"\n"); }
                else { writer.write(l1[i]+","+l2[i]+","+k[i]+","+perp[i]+"\n"); }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static void crossValidationWriterKatz(double[] l1, double[] l2, double[] perp, double[] error, int len) {

        try {
            FileWriter writer = new FileWriter("/Users/admin/Desktop/CV_NLP.txt", true);
            for (int i=0; i<len; i++) {
                if (i != len-1) { writer.write(l1[i]+","+l2[i]+","+perp[i]+error[i]+"\n"); }
                else { writer.write(l1[i]+","+l2[i]+","+perp[i]+error[i]+"\n"); }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        // Parse command line flags and arguments
        Map<String, String> argMap = CommandLineUtils
                .simpleCommandLineParser(args);

        // Set up default parameters and settings
        String basePath = ".";
        String model = "baseline";
        boolean verbose = false;

        // Update defaults using command line specifications

        // The path to the assignment data
        if (argMap.containsKey("-path")) {
            basePath = argMap.get("-path");
        }
        System.out.println("Using base path: " + basePath);

        // A string descriptor of the model to use
        if (argMap.containsKey("-model")) {
            model = argMap.get("-model");
        }
        System.out.println("Using model: " + model);

        // Whether or not to print the individual speech errors.
        if (argMap.containsKey("-verbose")) {
            verbose = true;
        }
        if (argMap.containsKey("-quiet")) {
            verbose = false;
        }

        // Read in all the assignment data
        String trainingSentencesFile = "/treebank-sentences-spoken-train.txt";
        String validationSentencesFile = "/treebank-sentences-spoken-validate.txt";
        String speechNBestListsPath = "/wsj_n_bst";
        Collection<List<String>> trainingSentenceCollection = SentenceCollection.Reader
                .readSentenceCollection(basePath + trainingSentencesFile);
        Collection<List<String>> validationSentenceCollection = SentenceCollection.Reader
                .readSentenceCollection(basePath + validationSentencesFile);
        Set<String> trainingVocabulary = extractVocabulary(trainingSentenceCollection);
        Set<String> validationVocabulary = extractVocabulary(validationSentenceCollection);
        List<SpeechNBestList> speechNBestLists = SpeechNBestList.Reader
                .readSpeechNBestLists(basePath + speechNBestListsPath,
                        trainingVocabulary);

        // String validationSentencesFile =
        // "/treebank-sentences-spoken-validate.txt";
        // Collection<List<String>> validationSentenceCollection =
        // SentenceCollection.Reader.readSentenceCollection(basePath +
        // validationSentencesFile);

        // String testSentencesFile = "/treebank-sentences-spoken-test.txt";
        // Collection<List<String>> testSentenceCollection =
        // SentenceCollection.Reader.readSentenceCollection(basePath +
        // testSentencesFile);

        // Build the language model
        LanguageModel languageModel = null;
        if (model.equalsIgnoreCase("baseline")) {
            languageModel = new EmpiricalUnigramLanguageModel(
                    trainingSentenceCollection);
        } else if (model.equalsIgnoreCase("sri")) {
            languageModel = new SriLanguageModel(argMap.get("-sri"));
        } else if (model.equalsIgnoreCase("bigram")) {
            languageModel = new EmpiricalBigramLanguageModel(
                    trainingSentenceCollection);
        } else if (model.equalsIgnoreCase("trigram")) {
            languageModel = new EmpiricalTrigramLanguageModel(
                    trainingSentenceCollection);
        } else if (model.equalsIgnoreCase("katz-bigram")) {
            languageModel = new KatzBigramLanguageModel(
                    trainingSentenceCollection);
        } else if (model.equalsIgnoreCase("katz-trigram")) {
            double lambda1;
            double lambda2;
            int ind = 0;
            double minPerp = Double.MAX_VALUE;
            int minInd = -1;
            double[] perpArray = new double[2500];
            double[] lambda1Array = new double[2500];
            double[] lambda2Array = new double[2500];
            double[] wordErrorArray = new double[2500];

            for (int i=1; i<100; i+=2) {
                for (int j=0; j<100-i; j+=2) {
                        lambda1 = i * 0.01;
                        lambda2 = j * 0.01;
                        languageModel = new KatzTrigramLanguageModel(trainingSentenceCollection, lambda1, lambda2);
                        perpArray[ind] = calculatePerplexity(languageModel, validationSentenceCollection);
                        wordErrorArray[ind] = calculateWordErrorRate(languageModel, speechNBestLists, verbose);
                        System.out.println("lambda1: "+lambda1+", lambda2: "+lambda2+", HUB WER: "+wordErrorArray[ind]);
                        lambda1Array[ind] = lambda1;
                        lambda2Array[ind] = lambda2;
                        if (perpArray[ind] < minPerp) {
                            minPerp = perpArray[ind];
                            minInd = ind;
                        }
                        ind++;
                }
            }

            //double[] bestPara = retrieveParameterFromInd(minInd);
            lambda1 = lambda1Array[minInd];
            lambda2 = lambda2Array[minInd];
            System.out.println("Best paprameters: lambda1 -> "+lambda1+"; lambda2 -> "+lambda2);
            languageModel = new KatzTrigramLanguageModel(trainingSentenceCollection, lambda1, lambda2);
            crossValidationWriterKatz(lambda1Array, lambda2Array, perpArray, wordErrorArray, 45);
        } else if (model.equalsIgnoreCase("katz-trigram-wz")) {
            double lambda1;
            double lambda2;
            int K;
            int ind = 0;
            double minPerp = Double.MAX_VALUE;
            int minInd = -1;
            double[] perpArray = new double[495];
            double[] lambda1Array = new double[495];
            double[] lambda2Array = new double[495];
            double[] KArray = new double[495];
            double[] wordErrorArray = new double[495];

            for (int i=1; i<10; i++) {
                for (int j=0; j<10-i; j++) {
                    for (int k=10; k<=60; k+=5) {
                        lambda1 = i * 0.1;
                        lambda2 = j * 0.1;
                        K = k;
                        languageModel = new KatzTrigramLanguageModelWZ(trainingSentenceCollection, lambda1, lambda2, K);
                        perpArray[ind] = calculatePerplexity(languageModel, validationSentenceCollection);
                        wordErrorArray[ind] = calculateWordErrorRate(languageModel, speechNBestLists, verbose);
                        System.out.println("lambda1: "+lambda1+", lambda2: "+lambda2+", K: "+K+", HUB WER: "+wordErrorArray[ind]);
                        lambda1Array[ind] = lambda1;
                        lambda2Array[ind] = lambda2;
                        KArray[ind] = (double) K;
                        if (perpArray[ind] < minPerp) {
                            minPerp = perpArray[ind];
                            minInd = ind;
                        }
                        ind++;
                    }
                }
            }

            //double[] bestPara = retrieveParameterFromInd(minInd);
            lambda1 = lambda1Array[minInd];
            lambda2 = lambda2Array[minInd];
            K = (int) KArray[minInd];
            System.out.println("Best paprameters: lambda1 -> "+lambda1+"; lambda2 -> "+lambda2+"; K -> "+K);
            languageModel = new KatzTrigramLanguageModelWZ(trainingSentenceCollection, lambda1, lambda2, K);
            crossValidationWriter(lambda1Array, lambda2Array, KArray, perpArray, 495);
        } else {
            throw new RuntimeException("Unknown model descriptor: " + model);
        }

        // Evaluate the language model
        // double wsjPerplexity = calculatePerplexity(languageModel,
        // testSentenceCollection);
        double hubPerplexity = calculatePerplexity(languageModel,
                extractCorrectSentenceList(speechNBestLists));

        double wsjPerplexity = calculatePerplexity(languageModel, validationSentenceCollection);

        System.out.println("WSJ Perplexity:  " + wsjPerplexity);
        System.out.println("HUB Perplexity:  " + hubPerplexity);
        System.out.println("WER Baselines:");
        System.out.println("  Best Path:  "
                + calculateWordErrorRateLowerBound(speechNBestLists));
        System.out.println("  Worst Path: "
                + calculateWordErrorRateUpperBound(speechNBestLists));
        System.out.println("  Avg Path:   "
                + calculateWordErrorRateRandomChoice(speechNBestLists));
        double wordErrorRate = calculateWordErrorRate(languageModel,
                speechNBestLists, verbose);
        System.out.println("HUB Word Error Rate: " + wordErrorRate);
        System.out.println("Generated Sentences:");
        // for (int i = 0; i < 10; i++)
        // System.out.println("  " + languageModel.generateSentence());
    }
}