package org.dora.algorithm.leetcode;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/12
 */
public class WordDictionary {

    private TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public WordDictionary() {
        root = new TrieNode();

    }

    public static void main(String[] args) {
        WordDictionary wordDictionary = new WordDictionary();
        wordDictionary.addWord("at");
        wordDictionary.addWord("and");
        wordDictionary.addWord("an");
        wordDictionary.addWord("add");
        wordDictionary.addWord("a");
    }

    /**
     * Adds a word into the data structure.
     */
    public void addWord(String word) {
        if (word == null || word.isEmpty()) {
            return;
        }
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
     * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        TrieNode p = root;
        return intervalSearch(0, word, p);
    }

    private boolean intervalSearch(int start, String word, TrieNode root) {
        if (start == word.length()) {
            return !"".equals(root.word);
        }
        if (word.charAt(start) != '.') {
            return root.nodes[word.charAt(start) - 'a'] != null &&
                    this.intervalSearch(start + 1, word, root.nodes[word.charAt(start) - 'a']);
        } else {
            for (int i = 0; i < root.nodes.length; i++) {
                if (root.nodes[i] != null && this.intervalSearch(start + 1, word, root.nodes[i])) {
                    return true;
                }
            }
        }
        return false;
    }

    static class TrieNode {
        private TrieNode[] nodes;
        private String word = "";

        public TrieNode() {
            nodes = new TrieNode[26];
        }
    }
}
