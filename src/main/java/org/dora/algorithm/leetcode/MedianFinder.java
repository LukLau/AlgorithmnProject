package org.dora.algorithm.leetcode;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/21
 */
public class MedianFinder {

    private PriorityQueue<Long> large = new PriorityQueue<>();

    private PriorityQueue<Long> small = new PriorityQueue<>(Comparator.reverseOrder());

    /**
     * initialize your data structure here.
     */
    public MedianFinder() {
    }

    public static void main(String[] args) {
        MedianFinder medianFinder = new MedianFinder();
    }

    public void addNum(int num) {

        large.add((long) num);

        small.add(large.poll());

        if (small.size() > large.size()) {
            large.add(small.poll());
        }
    }

    public double findMedian() {
        return large.size() > small.size() ?
                large.peek() : (large.peek() + small.peek()) / 2.0;
    }


}
