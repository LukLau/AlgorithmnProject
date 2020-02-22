package org.dora.algorithm.swordoffer;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/2/21
 */
public class MediaSolution {

    private PriorityQueue<Integer> small = new PriorityQueue<>(Comparator.reverseOrder());

    private PriorityQueue<Integer> big = new PriorityQueue<>();

    public static void main(String[] args) {
        MediaSolution mediaSolution = new MediaSolution();
        mediaSolution.Insert(1);
        System.out.println(mediaSolution.GetMedian());

        mediaSolution.Insert(-1);

        System.out.println(mediaSolution.GetMedian());

    }

    public void Insert(Integer num) {

        small.add(num);

        big.add(small.poll());

        if (big.size() > small.size()) {
            small.add(big.poll());
        }
    }

    public Double GetMedian() {
        if (big.size() < small.size()) {
            return small.peek() / 1.0;
        }
        return (big.peek() + small.peek()) / 2.0;
    }
}
