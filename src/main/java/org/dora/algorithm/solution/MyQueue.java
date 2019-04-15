package org.dora.algorithm.solution;

import java.util.Stack;

/**
 * @author liulu
 * @date 2019-04-07
 */
public class MyQueue {
    private Stack<Integer> push;
    private Stack<Integer> pop;

    /**
     * Initialize your data structure here.
     */
    public MyQueue() {
        push = new Stack<>();
        pop = new Stack<>();
    }

    /**
     * Push element x to the back of queue.
     */
    public void push(int x) {
        push.push(x);
    }

    /**
     * Removes the element from in front of queue and returns that element.
     */
    public int pop() {
        if (empty()) {
            return -1;
        }
        if (!pop.isEmpty()) {
            int top = pop.pop();
            return top;
        }
        while (!push.isEmpty()) {
            int top = push.pop();
            pop.push(top);
        }
        return pop.pop();
    }

    /**
     * Get the front element.
     */
    public int peek() {
        if (empty()) {
            return -1;
        }
        if (!pop.isEmpty()) {
            return pop.peek();
        }
        while (!push.isEmpty()) {
            int top = push.pop();
            pop.push(top);
        }
        return pop.peek();
    }

    /**
     * Returns whether the queue is empty.
     */
    public boolean empty() {
        return push.isEmpty() && pop.isEmpty();
    }

}
