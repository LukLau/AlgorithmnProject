package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-03-16
 */
public class Trie {

    TrieNode root;

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
        p.hasWord = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.nodes[word.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.nodes[word.charAt(i) - 'a'];
        }
        return p.hasWord;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode p = root;
        for (int i = 0; i < prefix.length(); i++) {
            if (p.nodes[prefix.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.nodes[prefix.charAt(i) - 'a'];
        }
        return true;
    }

    class TrieNode {
        private TrieNode[] nodes;
        private boolean hasWord;

        public TrieNode() {
            nodes = new TrieNode[26];
        }
    }
}
