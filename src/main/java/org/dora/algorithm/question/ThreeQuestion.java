package org.dora.algorithm.question;

import org.dora.algorithm.datastructe.ListNode;

import java.util.HashSet;
import java.util.Set;

/**
 * @author liulu
 * @date 2019-04-05
 */
public class ThreeQuestion {

    /**
     * 201、Bitwise AND of Numbers Range
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        return n > m ? rangeBitwiseAnd(m, n & n - 1) : n;
    }

    /**
     * 202. Happy Number
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        if (n <= 0) {
            return false;
        }
        Set<Integer> used = new HashSet<>();
        while (n != 0) {
            int result = 0;
            while (n != 0) {
                int index = n % 10;
                result += index * index;
                n /= 10;
            }
            if (result == 1) {
                return true;
            }
            if (used.contains(result)) {
                return false;
            }
            used.add(result);
            n = result;
        }
        return false;
    }

    /**
     * 203. Remove Linked List Elements
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode dummy = head;
        ListNode node = root;
        while (dummy != null) {
            if (dummy.val == val) {
                node.next = dummy.next;
            } else {
                node = node.next;
            }
            dummy = dummy.next;
        }
        return root.next;
    }

    /**
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        int[] hash1 = new int[256];
        int[] hash2 = new int[256];
        for (int i = 0; i < m; i++) {
            if (hash1[s.charAt(i)] != hash2[t.charAt(i)]) {
                return false;
            }
            hash1[s.charAt(i)] = i + 1;
            hash2[t.charAt(i)] = i + 1;
        }
        return true;
    }

    /**
     * 207. Course Schedule
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        return false;
    }
}
