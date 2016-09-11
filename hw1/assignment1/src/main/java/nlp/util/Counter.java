package nlp.util;

import java.io.Serializable;
import java.util.*;
import java.util.Map.Entry;
import java.lang.Math;


/**
 * A map from objects to doubles. Includes convenience methods for getting,
 * setting, and incrementing element counts. Objects not in the counter will
 * return a count of zero. The counter is backed by a HashMap (unless specified
 * otherwise with the MapFactory constructor).
 */
public class Counter<E> implements Serializable {
	private static final long serialVersionUID = 5724671156522771655L;

	Map<E, Double> entries;
	Map<E, Double> entriesKatz;

	int currentModCount = 0;
	int cacheModCount = -1;
	double cacheTotalCount = 0.0;

	/**
	 * The elements in the counter.
	 * 
	 * @return set of keys
	 */
	public Set<E> keySet() {
		return entries.keySet();
	}

	/**
	 * The number of entries in the counter (not the total count -- use
	 * totalCount() instead).
	 */
	public int size() {
		return entries.size();
	}

	/**
	 * True if there are no entries in the counter (false does not mean
	 * totalCount > 0)
	 */
	public boolean isEmpty() {
		return size() == 0;
	}

	/**
	 * Returns whether the counter contains the given key. Note that this is the
	 * way to distinguish keys which are in the counter with count zero, and
	 * those which are not in the counter (and will therefore return count zero
	 * from getCount().
	 * 
	 * @param key
	 * @return whether the counter contains the key
	 */
	public boolean containsKey(E key) {
		return entries.containsKey(key);
	}

	/**
	 * Remove a key from the counter. Returns the count associated with that key
	 * or zero if the key wasn't in the counter to begin with
	 * 
	 * @param key
	 * @return the count associated with the key
	 */
	public double removeKey(E key) {
		Double d = entries.remove(key);
		return (d == null ? 0.0 : d);
	}

	/**
	 * Get the count of the element, or zero if the element is not in the
	 * counter.
	 * 
	 * @param key
	 * @return
	 */
	public double getCount(E key) {
		Double value = entries.get(key);
		if (value == null)
			return 0;
		return value;
	}

//	public double getCountKatz(E key) {
//		Double value = entriesKatz.get(key);
//		if (value == null)
//			return 0;
//		return value;
//	}

	/**
	 * Set the count for the given key, clobbering any previous count.
	 * 
	 * @param key
	 * @param count
	 */
	public void setCount(E key, double count) {
		currentModCount++;
		entries.put(key, count);
	}

//	public void setCountKatz(E key, double count) {
//		currentModCount++;
//		entriesKatz.put(key, count);
//	}

	/**
	 * Increment a key's count by the given amount.
	 * 
	 * @param key
	 * @param increment
	 */
	public void incrementCount(E key, double increment) {
		setCount(key, getCount(key) + increment);
	}

	/**
	 * Increment each element in a given collection by a given amount.
	 */
	public void incrementAll(Collection<? extends E> collection, double count) {
		for (E key : collection) {
			incrementCount(key, count);
		}
	}

	public <T extends E> void incrementAll(Counter<T> counter) {
		for (T key : counter.keySet()) {
			double count = counter.getCount(key);
			incrementCount(key, count);
		}
	}

	public <T extends E> void elementwiseMax(Counter<T> counter) {
		for (T key : counter.keySet()) {
			double count = counter.getCount(key);
			if (getCount(key) < count) {
				setCount(key, count);
			}
		}
	}

	/**
	 * Finds the total of all counts in the counter. This implementation uses
	 * cached count which may get out of sync if the entries map is modified in
	 * some unantipicated way.
	 * 
	 * @return the counter's total
	 */
	public double totalCount() {
		if (currentModCount != cacheModCount) {
			double total = 0.0;
			for (Map.Entry<E, Double> entry : entries.entrySet()) {
				total += entry.getValue();
			}
			cacheTotalCount = total;
			cacheModCount = currentModCount;
		}
		return cacheTotalCount;
	}

	/**
	 * Destructively normalize this Counter in place.
	 */
	public void normalize() {
		double totalCount = totalCount();
		for (E key : keySet()) {
			setCount(key, getCount(key) / totalCount);
		}
	}

	public void normalizeKatz(int K, CounterMap.logLinearModel lm, Counter<E> counterPrevious) {
		double tmpCount;
		double newCount;
		double adjusted;
		double discounted = 0.0;
		double sumPrviousZero = 0.0;
		double alpha = 0.0;
		double sumCurrentProb = 0.0;
		double totalCount = totalCount();

		for (E key : keySet()) {
			tmpCount = getCount(key);
			if (tmpCount > K) { setCount(key, tmpCount); }
			else if (tmpCount > 0) {
//				System.out.println("Size of count array: "+lm.adCountArray.length);
//				System.out.println("Current count: "+tmpCount);
				newCount = (tmpCount + 1) * (lm.adCountArray[(int) tmpCount] / lm.adCountArray[(int) tmpCount - 1]);
				if (newCount < tmpCount) {
					discounted += (tmpCount - newCount);
				}
				else { newCount = tmpCount; }
				setCount(key, newCount);
			}
			else {
				sumPrviousZero += counterPrevious.getCount(key);
				setCount(key, -1.0);
			}
		}

		if (sumPrviousZero > 0.0) {
			alpha = discounted / totalCount / sumPrviousZero;
		}

		for (E key : keySet()) {
			tmpCount = getCount(key);
			if (tmpCount < 0.0) { setCount(key, alpha * counterPrevious.getCount(key) ); }
		}

		for (E key : keySet()) {
			setCount(key, getCount(key) / totalCount);
		}
	}


	public logLinearModel fitLogLinearModel(int K) {
		return new logLinearModel(K);
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
			for (E key : keySet()) {
				tmpCount = getCount(key);
				if (tmpCount <= K) {
					countArray[(int) tmpCount]++;
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
	 * Destructively scale this Counter in place.
	 */
	public void scale(double scaleFactor) {
		for (E key : keySet()) {
			setCount(key, getCount(key) * scaleFactor);
		}
	}

	/**
	 * Finds the key with maximum count. This is a linear operation, and ties
	 * are broken arbitrarily.
	 * 
	 * @return a key with minumum count
	 */
	public E argMax() {
		double maxCount = Double.NEGATIVE_INFINITY;
		E maxKey = null;
		for (Map.Entry<E, Double> entry : entries.entrySet()) {
			if (entry.getValue() > maxCount || maxKey == null) {
				maxKey = entry.getKey();
				maxCount = entry.getValue();
			}
		}
		return maxKey;
	}

	/**
	 * Returns a string representation with the keys ordered by decreasing
	 * counts.
	 * 
	 * @return string representation
	 */
	public String toString() {
		return toString(keySet().size());
	}

	/**
	 * Returns a string representation which includes no more than the
	 * maxKeysToPrint elements with largest counts.
	 * 
	 * @param maxKeysToPrint
	 * @return partial string representation
	 */
	public String toString(int maxKeysToPrint) {
		return asPriorityQueue().toString(maxKeysToPrint);
	}

	/**
	 * Builds a priority queue whose elements are the counter's elements, and
	 * whose priorities are those elements' counts in the counter.
	 */
	public PriorityQueue<E> asPriorityQueue() {
		PriorityQueue<E> pq = new FastPriorityQueue<E>(entries.size());
		for (Map.Entry<E, Double> entry : entries.entrySet()) {
			pq.setPriority(entry.getKey(), entry.getValue());
		}
		return pq;
	}

	/**
	 * Entry sets are an efficient way to iterate over the key-value pairs in a
	 * map
	 * 
	 * @return entrySet
	 */
	public Set<Entry<E, Double>> getEntrySet() {
		return entries.entrySet();
	}

	public Counter() {
		this(new MapFactory.HashMapFactory<E, Double>());
	}

	public Counter(MapFactory<E, Double> mf) {
		entries = mf.buildMap();
	}

	public Counter(Counter<? extends E> counter) {
		this();
		incrementAll(counter);
	}

	public Counter(Collection<? extends E> collection) {
		this();
		incrementAll(collection, 1.0);
	}

	public static void main(String[] args) {
		Counter<String> counter = new Counter<String>();
		System.out.println(counter);
		counter.incrementCount("planets", 7);
		System.out.println(counter);
		counter.incrementCount("planets", 1);
		System.out.println(counter);
		counter.setCount("suns", 1);
		System.out.println(counter);
		counter.setCount("aliens", 0);
		System.out.println(counter);
		System.out.println(counter.toString(2));
		System.out.println("Total: " + counter.totalCount());
	}

}
