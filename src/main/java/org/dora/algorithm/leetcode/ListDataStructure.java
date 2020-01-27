package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.ListNode;

/**
 * @author dora
 * @date 2019/11/6
 */
public class ListDataStructure {


    /**
     * 82. Remove Duplicates from Sorted List II
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode nextNode = head.next.next;
            while (nextNode != null && nextNode.val == head.val) {
                nextNode = nextNode.next;
            }
            return this.deleteDuplicates(nextNode);
        } else {
            head.next = this.deleteDuplicates(head.next);
            return head;
        }
    }


    /**
     * 86. Partition List
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode smallHead = new ListNode(0);
        ListNode bigHead = new ListNode(0);

        ListNode dummy1 = smallHead;

        ListNode dummy2 = bigHead;

        while (head != null) {
            if (head.val < x) {
                dummy1.next = head;
                dummy1 = dummy1.next;
            } else {
                dummy2.next = head;

                dummy2 = dummy2.next;
            }
            head = head.next;
        }
        // 这样代码必须添加 否则会导致 链表未拆分
        dummy2.next = null;

        ListNode next = bigHead.next;

        dummy1.next = next;

        bigHead.next = null;

        return smallHead.next;

    }


    /**
     * 92. Reverse Linked List II
     *
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode slow = root;

        ListNode fast = root;

        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }

        ListNode prev = slow.next;

        ListNode end = fast.next;

        ListNode node = this.reverseListNode(prev, fast);

        slow.next = node;

        prev.next = end;

        return root.next;

    }


    private ListNode reverseListNode(ListNode start, ListNode end) {
        ListNode prev = null;

        while (prev != end) {
            ListNode tmp = start.next;

            start.next = prev;

            prev = start;

            start = tmp;
        }
        return prev;
    }
}
