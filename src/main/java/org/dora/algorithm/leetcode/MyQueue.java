package org.dora.algorithm.leetcode;

import java.util.Stack;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/17
 */
public class MyQueue {


    private Stack<Integer> pushStack;
    private Stack<Integer> popStack;

    /**
     * Initialize your data structure here.
     */
    public MyQueue() {
        pushStack = new Stack<>();
        popStack = new Stack<>();
    }

    /**
     * Push element x to the back of queue.
     */
    public void push(int x) {
        pushStack.push(x);
    }

    /**
     * Removes the element from in front of queue and returns that element.
     */
    public int pop() {
        if (this.empty()) {
            return -1;
        }
        if (!popStack.isEmpty()) {
            return popStack.pop();
        }
        while (!pushStack.isEmpty()) {
            Integer pop = pushStack.pop();
            popStack.push(pop);
        }
        return popStack.pop();
    }

    /**
     * Get the front element.
     */
    public int peek() {
        if (this.empty()) {
            return -1;
        }
        if (!popStack.isEmpty()) {
            return popStack.peek();
        }
        while (!pushStack.isEmpty()) {
            Integer pop = pushStack.pop();
            popStack.push(pop);
        }
        return popStack.peek();
    }

    /**
     * Returns whether the queue is empty.
     */
    public boolean empty() {
        return pushStack.isEmpty() && popStack.isEmpty();
    }
}
