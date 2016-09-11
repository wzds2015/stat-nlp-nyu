package nlp.util;

import java.util.*;

/**
 * Maintains counts of (key, value) pairs. The map is structured so that for
 * every key, one can get a counter over values. Example usage: keys might be
 * words with values being POS tags, and the count being the number of
 * occurrences of that word/tag pair. The sub-counters returned by
 * getCounter(word) would be count distributions over tags for that word.
 */
public class CounterMap<K, V> implements java.io.Serializable {
	private static final long serialVersionUID = 5724671156522771668L;

	MapFactory<V, Double> mf;
	public Map<K, Counter<V>> counterMap;

	int currentModCount = 0;
	int cacheModCount = -1;
	double cacheTotalCount = 0.0;

	protected Counter<V> ensureCounter(K key) {
		Counter<V> valueCounter = counterMap.get(key);
		if (valueCounter == null) {
			valueCounter = new Counter<V>(mf);
			counterMap.put(key, valueCounter);
		}
		return valueCounter;
	}

	/**
	 * Returns the keys that have been inserted into this CounterMap.
	 */
	public Set<K> keySet() {
		return counterMap.keySet();
	}

	/**
	 * Sets the count for a particular (key, value) pair.
	 */
	public void setCount(K key, V value, double count) {
		Counter<V> valueCounter = ensureCounter(key);
		valueCounter.setCount(value, count);
		currentModCount++;
	}

//	public void setCountKatz(K key, V value, double count) {
//		Counter<V> valueCounter = ensureCounter(key);
//		valueCounter.setCountKatz(value, count);
//		currentModCount++;
//	}

	/**
	 * Increments the count for a particular (key, value) pair.
	 */
	public void incrementCount(K key, V value, double count) {
		Counter<V> valueCounter = ensureCounter(key);
		valueCounter.incrementCount(value, count);
		currentModCount++;
	}

	public void incrementAll(Map<K, V> map, double count) {
		for (Map.Entry<K, V> entry : map.entrySet()) {
			incrementCount(entry.getKey(), entry.getValue(), count);
		}
	}

	public void incrementAll(Collection<Pair<K, V>> entries, double count) {
		for (Pair<K, V> entry : entries) {
			incrementCount(entry.getFirst(), entry.getSecond(), count);
		}
	}

	/**
	 * Gets the count of the given (key, value) entry, or zero if that entry is
	 * not present. Does not create any objects.
	 */
	public double getCount(K key, V value) {
		Counter<V> valueCounter = counterMap.get(key);
		if (valueCounter == null)
			return 0.0;
		return valueCounter.getCount(value);
	}

//	public double getCountKatz(K key, V value) {
//		Counter<V> valueCounter = counterMap.get(key);
//		if (valueCounter == null)
//			return 0.0;
//		return valueCounter.getCountKatz(value);
//	}

	/**
	 * Gets the sub-counter for the given key. If there is none, a counter is
	 * created for that key, and installed in the CounterMap. You can, for
	 * example, add to the returned empty counter directly (though you
	 * shouldn't). This is so whether the key is present or not, modifying the
	 * returned counter has the same effect (but don't do it).
	 */
	public Counter<V> getCounter(K key) {
		return ensureCounter(key);
	}

	/**
	 * Returns whether or not the <code>CounterMap</code> contains any entries
	 * for the given key.
	 * 
	 * @author Aria Haghighi
	 * @param key
	 * @return
	 */
	public boolean containsKey(K key) {
		return counterMap.containsKey(key);
	}

	/**
	 * Returns the total of all counts in sub-counters. This implementation is
	 * caches the result -- it can get out of sync if the entries get modified
	 * externally.
	 */
	public double totalCount() {
		if (currentModCount != cacheModCount) {
			double total = 0.0;
			for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
				Counter<V> counter = entry.getValue();
				total += counter.totalCount();
			}
			cacheTotalCount = total;
			cacheModCount = currentModCount;
		}
		return cacheTotalCount;
	}

	/**
	 * Returns the total number of (key, value) entries in the CounterMap (not
	 * their total counts).
	 */
	public int totalSize() {
		int total = 0;
		for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
			Counter<V> counter = entry.getValue();
			total += counter.size();
		}
		return total;
	}

	public CounterMap.logLinearModel fitLogLinearModel(int K) {
		return new CounterMap.logLinearModel(K);
	}

	public class logLinearModel {
		double slope;
		double intersect;
		double[] adCountArray;

		public logLinearModel(double K) {
			double[] countArray = new double[(int) K+1];
			ArrayList<Integer> indArray1 = new ArrayList<Integer>();
			ArrayList<Double> countArray1 = new ArrayList<Double>();
			for (int i=0; i<(int)K+1; i++) {
				countArray[i] = 0.0;
			}

			double tmpCount;
			for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
				Counter<V> counter = entry.getValue();
				for (V key : counter.keySet()) {
					tmpCount = counter.getCount(key);
					if (tmpCount <= K) {
						countArray[(int) tmpCount]++;
					}
				}
			}

			for (int i=0; i<(int)K+1; i++) {
				if (countArray[i] > 0.1) {
					indArray1.add(i);
					countArray1.add(Math.log(countArray[i]) );
				}
			}
			double[] coeffs = linearRegression(indArray1, countArray1);

			slope = coeffs[0];
			intersect = coeffs[1];

			adCountArray = new double[(int) K+1];
			Iterator<Integer> iteratorInd = indArray1.iterator();
			int i = 0;
			while (iteratorInd.hasNext()) {
				adCountArray[i] = Math.exp(slope * iteratorInd.next() + intersect);
				i++;
			}
		}
	}

	public double[] linearRegression(ArrayList ind, ArrayList count) {
		int n = ind.size();
		double[] x = new double[n];
		double[] y = new double[n];
		Iterator<Integer> iteratorX = ind.iterator();
		Iterator<Double> iteratorY = count.iterator();
		int i = 0;
		while (iteratorX.hasNext()) {
			x[i] = iteratorX.next();
			y[i] = iteratorY.next();
			i++;
		}
		double sumx = 0.0, sumy = 0.0, sumx2 = 0.0;
		for (i=0; i<n; i++) {
			sumx  += x[i];
			sumx2 += x[i] * x[i];
			sumy  += y[i];
		}
		double xbar = sumx / n;
		double ybar = sumy / n;

		// second pass: compute summary statistics
		double xxbar = 0.0, yybar = 0.0, xybar = 0.0;
		for (i = 0; i < n; i++) {
			xxbar += (x[i] - xbar) * (x[i] - xbar);
			yybar += (y[i] - ybar) * (y[i] - ybar);
			xybar += (x[i] - xbar) * (y[i] - ybar);
		}
		double beta1 = xybar / xxbar;
		double beta0 = ybar - beta1 * xbar;

		double[] res = {beta1, beta0};
		return res;
	}


	/**
	 * Normalizes the maps inside this CounterMap -- not the CounterMap itself.
	 */
//	public void normalize() {
//		for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
//			Counter<V> counter = entry.getValue();
//			counter.normalize();
//		}
//		currentModCount++;
//	}

	public void normalize(int K, CounterMap.logLinearModel lm, Counter<V> counterPrevious) {
		double subTotalCount;
		double tmpCount;
		double newCount;
		double discounted = 0.0;
		double sumPrviousZero = 0.0;
		double alpha = 0.0;
		double sumCurrent = 0.0;


		for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
			Counter<V> counter = entry.getValue();
			subTotalCount = counter.totalCount();
			for (V key : counter.keySet()) {
				tmpCount = counter.getCount(key);
				if (tmpCount > K) {
					counter.setCount(key, tmpCount);
				} else if (tmpCount > 0) {
					newCount = (tmpCount + 1) * (lm.adCountArray[(int) tmpCount] / lm.adCountArray[(int) tmpCount - 1]);
					if (newCount < tmpCount) {
						discounted += (tmpCount - newCount);
					} else {
						newCount = tmpCount;
					}
					counter.setCount(key, newCount);
				} else {
					sumPrviousZero += counterPrevious.getCount(key);
					counter.setCount(key, -1.0);
				}
			}

			if (sumPrviousZero > 0.0) {
				alpha = discounted / subTotalCount / sumPrviousZero;
			}

			for (V key : counter.keySet()) {
				tmpCount = counter.getCount(key);
				if (tmpCount < 0.0) {
					counter.setCount(key, alpha * counterPrevious.getCount(key));
				}
			}

			counter.normalize();
		}
	}


	/**
	 * The number of keys in this CounterMap (not the number of key-value
	 * entries -- use totalSize() for that)
	 */
	public int size() {
		return counterMap.size();
	}

	/**
	 * True if there are no entries in the CounterMap (false does not mean
	 * totalCount > 0)
	 */
	public boolean isEmpty() {
		return size() == 0;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder("[\n");
		for (Map.Entry<K, Counter<V>> entry : counterMap.entrySet()) {
			sb.append("  ");
			sb.append(entry.getKey());
			sb.append(" -> ");
			sb.append(entry.getValue());
			sb.append("\n");
		}
		sb.append("]");
		return sb.toString();
	}

	public CounterMap() {
		this(new MapFactory.HashMapFactory<K, Counter<V>>(),
				new MapFactory.HashMapFactory<V, Double>());
	}

	public CounterMap(MapFactory<K, Counter<V>> outerMF,
			MapFactory<V, Double> innerMF) {
		mf = innerMF;
		counterMap = outerMF.buildMap();
	}

	public static void main(String[] args) {
		CounterMap<String, String> bigramCounterMap = new CounterMap<String, String>();
		bigramCounterMap.incrementCount("people", "run", 1);
		bigramCounterMap.incrementCount("cats", "growl", 2);
		bigramCounterMap.incrementCount("cats", "scamper", 3);
		System.out.println(bigramCounterMap);
		System.out.println("Entries for cats: "
				+ bigramCounterMap.getCounter("cats"));
		System.out.println("Entries for dogs: "
				+ bigramCounterMap.getCounter("dogs"));
		System.out.println("Count of cats scamper: "
				+ bigramCounterMap.getCount("cats", "scamper"));
		System.out.println("Count of snakes slither: "
				+ bigramCounterMap.getCount("snakes", "slither"));
		System.out.println("Total size: " + bigramCounterMap.totalSize());
		System.out.println("Total count: " + bigramCounterMap.totalCount());
		System.out.println(bigramCounterMap);
	}
}
