package org.dora.algorithm.leetcode;

import java.util.Stack;

/**
 * @author dora
 * @date 2019/11/28
 */
public class MinStack {
    private Stack<Integer> valueStack;
    private Stack<Integer> minStack;

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        this.valueStack = new Stack<>();
        this.minStack = new Stack<>();
    }

    public void push(int x) {
        valueStack.add(x);
        if (!minStack.isEmpty()) {
            Integer peek = minStack.peek();
            if (peek >= x) {
                minStack.push(x);
            }
        } else {
            minStack.push(x);
        }
    }

    public void pop() {
        Integer pop = valueStack.pop();
        if (pop.equals(minStack.peek())) {
            minStack.pop();
        }
    }

    public int top() {
        return valueStack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }


}
