package org.dora.algorithm.leetcode;

import java.util.Iterator;
import java.util.List;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/29
 */
public class Vector2D implements Iterator<Integer> {
    private Iterator<List<Integer>> rowIterator;
    private Iterator<Integer> tmp;

    public Vector2D(List<List<Integer>> vec2d) {
        // Initialize your data structure here
        rowIterator = vec2d.iterator();
    }

    @Override
    public boolean hasNext() {
        // Write your code here
        while ((tmp == null || !tmp.hasNext()) && rowIterator.hasNext()) {
            tmp = rowIterator.next().iterator();
        }
        return tmp != null && tmp.hasNext();
    }

    @Override
    public Integer next() {
        if (!this.hasNext()) {
            return null;
        }
        return tmp.next();
    }

    @Override
    public void remove() {

    }
}
