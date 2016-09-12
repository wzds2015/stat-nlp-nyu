package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.MapFactory;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 */
class KatzTrigramLanguageModelWZ implements LanguageModel {

    static final String START = "<S>";
    static final String STOP = "</S>";
    static final String UNKNOWN = "*UNKNOWN*";
//    static final double lambda1 = 0.5;
//    static final double lambda2 = 0.3;
    double lambda1;
    double lambda2;

    Counter<String> wordCounter = new Counter<String>();
    CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
    CounterMap<String, String> trigramCounter = new CounterMap<String, String>();

    Counter.logLinearModel oneGramLM;
    CounterMap.logLinearModel biGramLM;
    CounterMap.logLinearModel triGramLM;

    public double getTrigramProbability(String prePreviousWord,
                                        String previousWord, String word) {
        double trigramCount = trigramCounter.getCount(prePreviousWord
                + previousWord, word);
        double bigramCount = bigramCounter.getCount(previousWord, word);
        double unigramCount = wordCounter.getCount(word);
        if (unigramCount == 0) {
            //System.out.println("UNKNOWN Word: " + word);
            unigramCount = wordCounter.getCount(UNKNOWN);
        }
        return lambda1 * trigramCount + lambda2 * bigramCount
                + (1.0 - lambda1 - lambda2) * unigramCount;
    }

    public double getSentenceProbability(List<String> sentence) {
        List<String> stoppedSentence = new ArrayList<String>(sentence);
        stoppedSentence.add(0, START);
        stoppedSentence.add(0, START);
        stoppedSentence.add(STOP);
        double probability = 1.0;
        String prePreviousWord = stoppedSentence.get(0);
        String previousWord = stoppedSentence.get(1);
        for (int i = 2; i < stoppedSentence.size(); i++) {
            String word = stoppedSentence.get(i);
            probability *= getTrigramProbability(prePreviousWord, previousWord,
                    word);
            prePreviousWord = previousWord;
            previousWord = word;
        }
        if (probability == 0)
            System.err.println("Underflow");
        return probability;
    }

    String generateWord() {
        double sample = Math.random();
        double sum = 0.0;
        for (String word : wordCounter.keySet()) {
            sum += wordCounter.getCount(word);
            if (sum > sample) {
                return word;
            }
        }
        return UNKNOWN;
    }

    public List<String> generateSentence() {
        List<String> sentence = new ArrayList<String>();
        String word = generateWord();
        while (!word.equals(STOP)) {
            sentence.add(word);
            word = generateWord();
        }
        return sentence;
    }

    public KatzTrigramLanguageModelWZ(Collection<List<String>> sentenceCollection,
                                      double l1, double l2, int K) {
        lambda1 = l1;
        lambda2 = l2;
        for (List<String> sentence : sentenceCollection) {
            List<String> stoppedSentence = new ArrayList<String>(sentence);
            stoppedSentence.add(0, START);
            stoppedSentence.add(0, START);
            stoppedSentence.add(STOP);
            String prePreviousWord = stoppedSentence.get(0);
            String previousWord = stoppedSentence.get(1);
            for (int i = 2; i < stoppedSentence.size(); i++) {
                String word = stoppedSentence.get(i);
                wordCounter.incrementCount(word, 1.0);
                bigramCounter.incrementCount(previousWord, word, 1.0);
                trigramCounter.incrementCount(prePreviousWord + previousWord,
                        word, 1.0);
                prePreviousWord = previousWord;
                previousWord = word;
            }
        }
        wordCounter.incrementCount(UNKNOWN, 1.0);
        normalizeDistributions(K);
    }

    public Counter<String> counterMapTpCounter(CounterMap<String, String> biGCounter) {
        Counter<String> con = new Counter<String>(new MapFactory.HashMapFactory<String, Double>());

        String newKey;
        for (Map.Entry<String, Counter<String>> entry : biGCounter.counterMap.entrySet()) {
            Counter<String> counter = entry.getValue();
            for (String key : counter.keySet()) {
                newKey = entry.getKey() + key;
                con.setCount(newKey, counter.getCount(key));
            }
        }

        return con;
    }

    private void normalizeDistributions(int K) {
        oneGramLM = wordCounter.fitLogLinearModel(K);
        wordCounter.normalize();

        biGramLM = bigramCounter.fitLogLinearModel(K);
        for (String previousWord : bigramCounter.keySet()) {
            bigramCounter.getCounter(previousWord).normalizeKatz(K, biGramLM, wordCounter);
        }

        triGramLM = trigramCounter.fitLogLinearModel(K);
        Counter<String> newBigramCounter = counterMapTpCounter(bigramCounter);
        for (String previousBigram : trigramCounter.keySet()) {
            trigramCounter.getCounter(previousBigram).normalizeKatz(K, triGramLM, newBigramCounter);
        }

    }

}

