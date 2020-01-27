package org.dora.algorithm.leetcode;

import java.util.LinkedList;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/14
 */
public class MyStack {

    private LinkedList<Integer> linkedList = new LinkedList<>();
    ;

    /**
     * Initialize your data structure here.
     */
    public MyStack() {
    }

    /**
     * Push element x onto stack.
     */
    public void push(int x) {
        linkedList.add(x);
        for (int i = 0; i < linkedList.size() - 1; i++) {
            Integer poll = linkedList.poll();
            linkedList.add(poll);
        }
    }

    /**
     * Removes the element on top of the stack and returns that element.
     */
    public int pop() {
        return linkedList.poll();
    }

    /**
     * Get the top element.
     */
    public int top() {
        return linkedList.peekFirst();
    }

    /**
     * Returns whether the stack is empty.
     */
    public boolean empty() {
        return linkedList.isEmpty();
    }
}
