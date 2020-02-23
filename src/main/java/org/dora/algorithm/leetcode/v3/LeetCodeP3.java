package org.dora.algorithm.leetcode.v3;

import java.util.*;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/2/23
 */
public class LeetCodeP3 {


    public static void main(String[] args) {
        LeetCodeP3 p3 = new LeetCodeP3();
        String test = "(()";
        p3.removeInvalidParentheses(test);
    }

    /**
     * 301. Remove Invalid Parentheses
     *
     * @param s
     * @return
     */
    public List<String> removeInvalidParentheses(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        Set<String> visited = new HashSet<>();
        List<String> result = new ArrayList<>();
        LinkedList<String> deque = new LinkedList<>();

        deque.offer(s);
        visited.add(s);

        while (!deque.isEmpty()) {
            String poll = deque.poll();

            if (intervalInvalid(poll)) {
                result.add(poll);
            }

            if (result.size() != 0) {
                continue;
            }

            for (int i = 0; i < poll.length(); i++) {
                char character = poll.charAt(i);
                if (character != '(' && character != ')') {
                    continue;
                }
                String tmp = poll.substring(0, i) + poll.substring(i + 1);

                if (!visited.contains(tmp)) {
                    deque.offer(tmp);
                    visited.add(tmp);
                }

            }
        }
        if (result.isEmpty()) {
            result.add("");
        }
        return result;
    }

    private boolean intervalInvalid(String poll) {
        if (poll == null || poll.isEmpty()) {
            return true;
        }
        int length = poll.length();

        int count = 0;

        for (int i = 0; i < length; i++) {
            char character = poll.charAt(i);
            if (character != '(' && character != ')') {
                continue;
            }
            if (character == '(') {
                count++;
            }
            if (character == ')') {
                if (count == 0) {
                    return false;
                }
                count--;
            }
        }
        return count == 0;
    }


    public List<String> removeInvalidParenthesesV2(String s) {


        List<String> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.add(s);
        visited.add(s);
        boolean found = false;
        while (!queue.isEmpty()) {
            String poll = queue.poll();

            if (checkValid(poll)) {
                result.add(poll);
                found = true;
            }

            if (found) {
                continue;
            }

            for (int i = 0; i < poll.length(); i++) {
                char c = poll.charAt(i);
                if (c != '(' && c != ')') {
                    continue;
                }
                String tmp = poll.substring(0, i) + poll.substring(i + 1);
                if (!visited.contains(tmp)) {
                    queue.offer(tmp);
                    visited.add(tmp);
                }
            }
        }
        return result;
    }

    private boolean checkValid(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c != '(' && c != ')') {
                continue;
            }
            if (c == '(') {
                count++;
            }
            if (c == ')' && count-- == 0) {
                return false;
            }
        }
        return count == 0;
    }
}
