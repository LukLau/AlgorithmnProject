package org.dora.algorithm.leetcode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/16
 */
public class ZigzagIterator {


    private Iterator<Integer> iterator;

    /*
     * @param v1: A 1d vector
     * @param v2: A 1d vector
     */
    public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
        // do intialization if necessary
        List<Integer> interval = new ArrayList<>();

        Iterator<Integer> iterator1 = v1.iterator();
        Iterator<Integer> iterator2 = v2.iterator();
        while (iterator1.hasNext() || iterator2.hasNext()) {
            if (iterator1.hasNext()) {
                interval.add(iterator1.next());
            }
            if (iterator2.hasNext()) {
                interval.add(iterator2.next());
            }
        }
        iterator = interval.iterator();
    }

    /*
     * @return: An integer
     */
    public int next() {
        // write your code here
        return iterator.next();
    }

    /*
     * @return: True if has next
     */
    public boolean hasNext() {
        // write your code here
        return iterator.hasNext();
    }

}
