package org.dora.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * @author dora
 * @date 2019/11/27
 */
public class LRUCache {

    private Node head;
    private Node tail;
    private Map<Integer, Node> map = new HashMap<>();
    private int capacity = 0;

    public LRUCache(int capacity) {

        this.capacity = capacity;

        head = new Node();

        tail = new Node();

        head.next = tail;

        tail.prev = head;

    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        this.moveNode(node);
        return node.value;
    }

    private void moveNode(Node node) {
        this.removeNode(node);
        this.addNode(node);
    }

    private void addNode(Node node) {
        Node next = head.next;

        next.prev = node;
        head.next = node;

        node.prev = head;
        node.next = next;
    }

    private void removeNode(Node node) {
        Node prev = node.prev;
        Node next = node.next;

        prev.next = next;
        next.prev = prev;

        node.prev = null;
        node.next = null;
    }

    private Node popLastNode() {
        Node lastNode = tail.prev;
        this.removeNode(lastNode);
        return lastNode;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.value = value;
            this.moveNode(node);
        } else {
            Node insertNode = new Node(key, value);
            map.put(key, insertNode);
            this.addNode(insertNode);
            if (map.size() > capacity) {
                Node lastNode = this.popLastNode();
                map.remove(lastNode.key);
            }
        }
    }


    class Node {
        private Integer value;
        private Integer key;
        private Node prev;
        private Node next;

        public Node(int value) {
            this.value = value;
        }

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }

        public Node() {
        }
    }
}
