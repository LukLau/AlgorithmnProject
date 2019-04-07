package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-03-16
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
            if (p.nexts[word.charAt(i) - 'a'] == null) {
                p.nexts[word.charAt(i) - 'a'] = new TrieNode();
            }
            p = p.nexts[word.charAt(i) - 'a'];
        }
        p.word = word;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.nexts[word.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.nexts[word.charAt(i) - 'a'];
        }
        return p.word.equals(word);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode p = root;
        for (int i = 0; i < prefix.length(); i++) {
            if (p.nexts[prefix.charAt(i) - 'a'] == null) {
                return false;
            }
            p = p.nexts[prefix.charAt(i) - 'a'];
        }
        return true;
    }

    class TrieNode {
        private TrieNode[] nexts;
        private String word;

        private TrieNode() {
            nexts = new TrieNode[26];
            word = "";
        }
    }
}
