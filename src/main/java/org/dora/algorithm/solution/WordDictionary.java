package org.dora.algorithm.solution;

/**
 * @author liulu
 * @date 2019-03-16
 */
public class WordDictionary {

    private TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public WordDictionary() {
        root = new TrieNode();
    }

    /**
     * Adds a word into the data structure.
     */
    public void addWord(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p.next[word.charAt(i) - 'a'] == null) {
                p.next[word.charAt(i) - 'a'] = new TrieNode();
            }
            p = p.next[word.charAt(i) - 'a'];
        }
        p.word = word;
    }

    /**
     * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {
        if (word.equals("")) {
            return false;
        }
        return search(0, word, root);
    }

    private boolean search(int i, String word, TrieNode root) {
        if (i == word.length()) {
            return !root.word.equals("");
        }
        if (word.charAt(i) != '.') {
            return root.next[word.charAt(i) - 'a'] != null && search(i + 1, word, root.next[word.charAt(i) - 'a']);
        } else {
            for (int k = 0; k < root.next.length; k++) {
                if (root.next[k] != null && search(i + 1, word, root.next[k])) {
                    return true;

                }
            }
            return false;
        }
    }

    class TrieNode {
        private TrieNode[] next;
        private String word;

        public TrieNode() {
            next = new TrieNode[26];
            word = "";
        }
    }
}
