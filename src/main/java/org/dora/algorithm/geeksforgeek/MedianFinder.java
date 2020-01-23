package org.dora.algorithm.geeksforgeek;

import java.util.PriorityQueue;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/21
 */
public class MedianFinder {

    private PriorityQueue<Integer> max = new PriorityQueue<>();
    private PriorityQueue<Integer> min = new PriorityQueue<>();

    /**
     * initialize your data structure here.
     */
    public MedianFinder() {
    }

    public static void main(String[] args) {

    }

    public void addNum(int num) {
        max.offer(num);
        min.offer(-max.poll());
        if (max.size() < min.size()) {
            max.offer(-min.poll());
        }
    }

    public double findMedian() {
        return max.size() > min.size()
                ? max.peek()
                : (max.peek() - min.peek()) / 2.0;
    }

}
