package org.dora.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/4
 */
public class TwoSum {

    private HashMap<Integer, Integer> map = new HashMap<>();

    /**
     * @param number: An integer
     * @return: nothing
     */
    public void add(int number) {
        Integer count = map.getOrDefault(number, 0);
        map.put(number, ++count);

        // write your code here
    }

    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     */
    public boolean find(int value) {
        Set<Map.Entry<Integer, Integer>> entries = map.entrySet();
        for (Map.Entry<Integer, Integer> entry : entries) {
            Integer key = entry.getKey();
            int remain = value - key;
            if (remain == key) {
                Integer count = entry.getValue();
                if (count <= 1) {
                    return false;
                }
                return true;
            } else if (map.containsKey(remain)) {
                return true;
            }
        }
        return false;
        // write your code here
    }

}
