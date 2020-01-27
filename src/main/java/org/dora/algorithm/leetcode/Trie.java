package org.dora.algorithm.leetcode;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/12
 */
public class Trie {
    private TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.nodes[word.charAt(i) - 'a'] == null) {
                p.nodes[word.charAt(i) - 'a'] = new TrieNode();
            }
            p = p.nodes[word.charAt(i) - 'a'];
        }
        p.word = word;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.nodes[word.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.nodes[word.charAt(i) - 'a'];
        }
        return word.equals(p.word);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        if (prefix == null || prefix.isEmpty()) {
            return false;
        }
        TrieNode p = root;

        for (int i = 0; i < prefix.length(); i++) {
            if (p.nodes[prefix.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.nodes[prefix.charAt(i) - 'a'];
        }
        return true;
    }


    static class TrieNode {
        private TrieNode[] nodes;

        private String word;

        public TrieNode() {
            nodes = new TrieNode[26];
        }
    }
}
