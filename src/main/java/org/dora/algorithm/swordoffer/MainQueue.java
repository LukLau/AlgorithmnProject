package org.dora.algorithm.swordoffer;

import java.util.Stack;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/29
 */
public class MainQueue {

    Stack<Integer> head = new Stack<>();
    Stack<Integer> tail = new Stack<>();

    public void push(int node) {
        head.push(node);
    }

    public int pop() {
        if (head.isEmpty() && tail.isEmpty()) {
            return -1;
        }
        if (!tail.isEmpty()) {
            return tail.pop();
        }
        while (!head.isEmpty()) {
            Integer pop = head.pop();
            tail.push(pop);
        }
        return tail.pop();
    }
}
