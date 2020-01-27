package org.dora.algorithm.leetcode;

import java.util.ArrayList;
import java.util.List;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/24
 */
public class MedianFinderV2 {


    private List<Long> ans = new ArrayList<>();

    /**
     * initialize your data structure here.
     */
    public MedianFinderV2() {

    }

    public void addNum(int num) {
        ans.add((long) num);
    }

    public double findMedian() {
        if (ans.isEmpty()) {
            return -1;
        }
        ans.sort(Long::compareTo);
        int len = ans.size();
        return (len & 1) != 0 ? ans.get(len / 2) / 1.0 : (ans.get(len / 2) + ans.get(len / 2 - 1)) / 2.0;
    }

}
